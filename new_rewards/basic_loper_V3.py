from tqdm import tqdm
import numpy as np
import re
import collections


from guesswhat.models.looper.tools import clear_after_stop_dialogue, list_to_padded_tokens


class BasicLooper(object):
    def __init__(self, config, oracle_wrapper, qgen_wrapper, guesser_wrapper, tokenizer, batch_size):
        self.storage = []

        self.tokenizer = tokenizer
        self.batch_size = batch_size

        self.max_no_question = config['loop']['max_question']
        self.max_depth = config['loop']['max_depth']
        self.k_best = config['loop']['beam_k_best']
        self.pen_spatial = config['loop']['pen_spatial']#added
        self.avoid_words = config['loop']['avoid_words']#added
        self.bonus_expl=config['loop']['bonus_exploration'] #added
        self.update_reward=config['loop']['update_reward']#added
        self.common_words=config['loop']['common_words']#added
        self.penalty_repetition=config['loop']['penalty_repetition'] #added
        self.current_batch=0
        self.oracle = oracle_wrapper
        self.guesser = guesser_wrapper
        self.qgen = qgen_wrapper
        self.word_counter = collections.Counter()
        self.current_word_counter=collections.Counter()
    

#EVOLUTION DU REWARD AVEC LE TEMPS POUR ENCOURAGER À L'UTILISATION DE NOUVEAUX MOTS
#paramètre time_update = 50 pour se demander tous les combien de batchsize le reward doit être ajournée
#On créé un compteur qui enregistre le nombre d'occurences de chaque mot

    def process(self, sess, iterator, mode, optimizer=list(), store_games=False):

        # initialize the wrapper
        self.qgen.initialize(sess)
        self.oracle.initialize(sess)
        self.guesser.initialize(sess)

        self.storage = []
        score, total_elem = 0, 0
        ### Here we eventually modify the word_counter, that will modify the reward function
        
        self.current_batch+=1
        if (self.current_batch % self.update_reward) ==0:
            for w in self.current_word_counter:
                self.word_counter[w]+=self.current_word_counter[w]
            self.current_word_counter=collections.Counter()
            print("<----reward upadted---->")
        
        
        for game_data in tqdm(iterator):

            # initialize the dialogue
            full_dialogues = [np.array([self.tokenizer.start_token]) for _ in range(self.batch_size)]
            prev_answers = full_dialogues

            prob_objects = []

            no_elem = len(game_data["raw"])
            total_elem += no_elem

            # Step 1: generate question/answer
            self.qgen.reset(batch_size=no_elem)
            for no_question in range(self.max_no_question):

                # Step 1.1: Generate new question
                padded_questions, questions, seq_length = \
                    self.qgen.sample_next_question(sess, prev_answers, game_data=game_data, mode=mode)

                # Step 1.2: Answer the question
                answers = self.oracle.answer_question(sess,
                                                      question=padded_questions,
                                                      seq_length=seq_length,
                                                      game_data=game_data)

                # Step 1.3: store the full dialogues
                for i in range(self.batch_size):
                    full_dialogues[i] = np.concatenate((full_dialogues[i], questions[i], [answers[i]]))

                # Step 1.4 set new input tokens
                prev_answers = [[a]for a in answers]

                # Step 1.5 Compute the probability of finding the object after each turn
                if store_games:
                    padded_dialogue, seq_length = list_to_padded_tokens(full_dialogues, self.tokenizer)
                    _, softmax, _ = self.guesser.find_object(sess, padded_dialogue, seq_length, game_data)
                    prob_objects.append(softmax)

                # Step 1.6 Check if all dialogues are stopped
                has_stop = True
                for d in full_dialogues:
                    has_stop &= self.tokenizer.stop_dialogue in d
                if has_stop:
                    break

            # Step 2 : clear question after <stop_dialogue>
            full_dialogues, _ = clear_after_stop_dialogue(full_dialogues, self.tokenizer)
            padded_dialogue, seq_length = list_to_padded_tokens(full_dialogues, self.tokenizer)

            # Step 3 : Find the object
            found_object, _, id_guess_objects = self.guesser.find_object(sess, padded_dialogue, seq_length, game_data)
            score += np.sum(found_object)
            
            exploration_reward=[]
            repetitions=[]
            for i, d in enumerate(full_dialogues) : 
                occs=collections.Counter()
                exploration_bonus=0.0
                n_words=0
                
                for w in d:
                   n_words+=1
                   self.current_word_counter[w]+=1
                   if not w in self.common_words:
                       occs[w]+=1
                   exploration_bonus+=1.0/(1.0*(self.word_counter[w]+1))
                exploration_bonus/=(1.0*(n_words+1.0))
                exploration_reward.append(exploration_bonus)
                repetition=0
                for w in occs:
                    repetition+=(occs[w]-1)
                repetitions.append(repetition)        
                            

            if store_games:
                prob_objects = np.transpose(prob_objects, axes=[1,0,2])
                for i, (d, g, t, f, go, po) in enumerate(zip(full_dialogues, game_data["raw"], game_data["targets_index"], found_object, id_guess_objects, prob_objects)):
                    self.storage.append({"dialogue": d,
                                         "game": g,
                                         "object_id": g.objects[t].id,
                                         "success": f,
                                         "guess_object_id": g.objects[go].id,
                                         "prob_objects" : po} )

            if len(optimizer) > 0:
                final_reward = [(int(found_object[i])+exploration_reward[i]*self.bonus_expl - repetitions[i]*self.penalty_repetition) for i in range(len(exploration_reward))] 

                self.apply_policy_gradient(sess,
                                           final_reward=final_reward,
                                           padded_dialogue=padded_dialogue,
                                           seq_length=seq_length,
                                           game_data=game_data,
                                           optimizer=optimizer)

        score = 1.0 * score / iterator.n_examples

        return score
 
    def process_old(self, sess, iterator, mode, optimizer=list(), store_games=False):

        # initialize the wrapper
        self.qgen.initialize(sess)
        self.oracle.initialize(sess)
        self.guesser.initialize(sess)

        self.storage = []
        score, total_elem = 0, 0
        for game_data in tqdm(iterator):

            # initialize the dialogue
            full_dialogues = [np.array([self.tokenizer.start_token]) for _ in range(self.batch_size)]
            prev_answers = full_dialogues

            prob_objects = []

            no_elem = len(game_data["raw"])
            total_elem += no_elem

            # Step 1: generate question/answer
            self.qgen.reset(batch_size=no_elem)
            for no_question in range(self.max_no_question):

                # Step 1.1: Generate new question
                padded_questions, questions, seq_length = \
                    self.qgen.sample_next_question(sess, prev_answers, game_data=game_data, mode=mode)

                # Step 1.2: Answer the question
                answers = self.oracle.answer_question(sess,
                                                      question=padded_questions,
                                                      seq_length=seq_length,
                                                      game_data=game_data)

                # Step 1.3: store the full dialogues
                for i in range(self.batch_size):
                    full_dialogues[i] = np.concatenate((full_dialogues[i], questions[i], [answers[i]]))

                # Step 1.4 set new input tokens
                prev_answers = [[a]for a in answers]

                # Step 1.5 Compute the probability of finding the object after each turn
                if store_games:
                    padded_dialogue, seq_length = list_to_padded_tokens(full_dialogues, self.tokenizer)
                    _, softmax, _ = self.guesser.find_object(sess, padded_dialogue, seq_length, game_data)
                    prob_objects.append(softmax)

                # Step 1.6 Check if all dialogues are stopped
                has_stop = True
                for d in full_dialogues:
                    has_stop &= self.tokenizer.stop_dialogue in d
                if has_stop:
                    break

            # Step 2 : clear question after <stop_dialogue>
            full_dialogues, _ = clear_after_stop_dialogue(full_dialogues, self.tokenizer)
            padded_dialogue, seq_length = list_to_padded_tokens(full_dialogues, self.tokenizer)

            # Step 3 : Find the object
            found_object, _, id_guess_objects = self.guesser.find_object(sess, padded_dialogue, seq_length, game_data)
            score += np.sum(found_object)

            if store_games:
                prob_objects = np.transpose(prob_objects, axes=[1,0,2])
                for i, (d, g, t, f, go, po) in enumerate(zip(full_dialogues, game_data["raw"], game_data["targets_index"], found_object, id_guess_objects, prob_objects)):
                    self.storage.append({"dialogue": d,
                                         "game": g,
                                         "object_id": g.objects[t].id,
                                         "success": f,
                                         "guess_object_id": g.objects[go].id,
                                         "prob_objects" : po} )

            if len(optimizer) > 0:
                final_reward = found_object + 0  # +1 if found otherwise 0

                self.apply_policy_gradient(sess,
                                           final_reward=final_reward,
                                           padded_dialogue=padded_dialogue,
                                           seq_length=seq_length,
                                           game_data=game_data,
                                           optimizer=optimizer)

        score = 1.0 * score / iterator.n_examples

        return score

    def process_penalty_spatial(self, sess, iterator, mode, optimizer=list(), store_games=False):

        # initialize the wrapper
        self.qgen.initialize(sess)
        self.oracle.initialize(sess)
        self.guesser.initialize(sess)

        self.storage = []
        score, total_elem = 0, 0

        for game_data in tqdm(iterator):

            # initialize the dialogue
            full_dialogues = [np.array([self.tokenizer.start_token]) for _ in range(self.batch_size)]
            prev_answers = full_dialogues

            prob_objects = []

            no_elem = len(game_data["raw"])
            total_elem += no_elem

            # Step 1: generate question/answer
            self.qgen.reset(batch_size=no_elem)
            for no_question in range(self.max_no_question):

                # Step 1.1: Generate new question
                padded_questions, questions, seq_length = \
                    self.qgen.sample_next_question(sess, prev_answers, game_data=game_data, mode=mode)

                # Step 1.2: Answer the question
                answers = self.oracle.answer_question(sess,
                                                      question=padded_questions,
                                                      seq_length=seq_length,
                                                      game_data=game_data)

                # Step 1.3: store the full dialogues
                for i in range(self.batch_size):
                    full_dialogues[i] = np.concatenate((full_dialogues[i], questions[i], [answers[i]]))

                # Step 1.4 set new input tokens
                prev_answers = [[a]for a in answers]

                # Step 1.5 Compute the probability of finding the object after each turn
                if store_games:
                    padded_dialogue, seq_length = list_to_padded_tokens(full_dialogues, self.tokenizer)
                    _, softmax, _ = self.guesser.find_object(sess, padded_dialogue, seq_length, game_data)
                    prob_objects.append(softmax)

                # Step 1.6 Check if all dialogues are stopped
                has_stop = True
                for d in full_dialogues:
                    has_stop &= self.tokenizer.stop_dialogue in d
                if has_stop:
                    break

            # Step 2 : clear question after <stop_dialogue>
            full_dialogues, _ = clear_after_stop_dialogue(full_dialogues, self.tokenizer)
            padded_dialogue, seq_length = list_to_padded_tokens(full_dialogues, self.tokenizer)

            # Step 3 : Find the object
            found_object, _, id_guess_objects = self.guesser.find_object(sess, padded_dialogue, seq_length, game_data)
            score += np.sum(found_object)
            
            nb_badwords=[]
            for i, d in enumerate(full_dialogues) : 
                nb_d=0
                for q in d.questions:
                    words = re.findall(r'\w+', q)
                    for w in words:
                        if(w.lower() in self.avoid_words):
                            nb_d+=1
                nb_badwords.append(nb_d)

            if store_games:
                prob_objects = np.transpose(prob_objects, axes=[1,0,2])
                for i, (d, g, t, f, go, po) in enumerate(zip(full_dialogues, game_data["raw"], game_data["targets_index"], found_object, id_guess_objects, prob_objects)):
                    self.storage.append({"dialogue": d,
                                         "game": g,
                                         "object_id": g.objects[t].id,
                                         "success": f,
                                         "guess_object_id": g.objects[go].id,
                                         "prob_objects" : po} )

            if len(optimizer) > 0:
                final_reward = [found_object[i]-nb_badwords[i]*self.pen_spatial for i in range(len(nb_badwords))] + 0  # +1 if found otherwise 0

                self.apply_policy_gradient(sess,
                                           final_reward=final_reward,
                                           padded_dialogue=padded_dialogue,
                                           seq_length=seq_length,
                                           game_data=game_data,
                                           optimizer=optimizer)

        score = 1.0 * score / iterator.n_examples

        return score
    
    
    def get_storage(self):
        return self.storage

    def apply_policy_gradient(self, sess, final_reward, padded_dialogue, seq_length, game_data, optimizer):

        # Compute cumulative reward TODO: move into an external function
        cum_rewards = np.zeros_like(padded_dialogue, dtype=np.float32)
        for i, (end_of_dialogue, r) in enumerate(zip(seq_length, final_reward)):
            cum_rewards[i, :(end_of_dialogue - 1)] = r  # gamma = 1

        # Create answer mask to ignore the reward for yes/no tokens
        answer_mask = np.ones_like(padded_dialogue)  # quick and dirty mask -> TODO to improve
        answer_mask[padded_dialogue == self.tokenizer.yes_token] = 0
        answer_mask[padded_dialogue == self.tokenizer.no_token] = 0
        answer_mask[padded_dialogue == self.tokenizer.non_applicable_token] = 0

        # Create padding mask to ignore the reward after <stop_dialogue>
        padding_mask = np.ones_like(padded_dialogue)
        padding_mask[padded_dialogue == self.tokenizer.padding_token] = 0
        # for i in range(np.max(seq_length)): print(cum_rewards[0][i], answer_mask[0][i],self.tokenizer.decode([padded_dialogue[0][i]]))

        # Step 4.4: optim step
        qgen = self.qgen.qgen  # retrieve qgen from wrapper (dirty)

        sess.run(optimizer,
                 feed_dict={
                     qgen.images: game_data["images"],
                     qgen.dialogues: padded_dialogue,
                     qgen.seq_length: seq_length,
                     qgen.padding_mask: padding_mask,
                     qgen.answer_mask: answer_mask,
                     qgen.cum_rewards: cum_rewards,
                 })

    
