import sys
sys.path.insert(0, 'src/')


from .llmvanilla import LLMVanilla
from service.modelservice import GenerationParameter
import pandas
from service.utils import evaluate
from .scoring import Score
from sklearn.model_selection import train_test_split
from .llmchain import LLMChain
import json, os
import random
import logging
from pathlib import Path
from .llmcache import LLMCache

def scorer_text(text):
    #return text

    newtext = "Q:"+text.split("Q:")[-1]    
    return newtext

def tempsave(label, response, score,name):
    return

class LLMCascade(object):
    def __init__(self, 
                 #service_names =['openaichat/gpt-3.5-turbo','openaichat/gpt-4'],
                 metric="em",
                 db_path= Path(__file__).parent.parent.parent / "db" / "AGNEWS.sqlite",
                 score_noise_injection=False,
                 batch_build = False,
                 prefix="",
                 use_cache=True,
                 cache_similarity_threshold=0.85
                 ):
        # Initialization code for the FrugalGPT class
        self.MyLLMEngine = LLMVanilla(db_path=db_path)    
        self.MyScores = dict()
        self.LLMChain = LLMChain(metric=metric)
        #self.service_names =['openaichat/gpt-3.5-turbo','openaichat/gpt-4'],
        self.eps=1e-8
        self.score_noise_injection = score_noise_injection
        self.batch_build = batch_build
        self.prefix = prefix
        self.use_cache = use_cache
        cache_file = "cache/query_cache.json"
        if use_cache:
            self.cache = LLMCache(
                cache_file,
                cache_similarity_threshold
            )
        return 

    def load(self,loadpath="strategy/HEADLINES/",budget=0.01):
        self.LLMChain = LLMChain()
        self.LLMChain.setbudget(budget=budget)
        self.LLMChain.loadstrategy(Path(__file__).parent.parent.parent  / "strategy" / "AGNEWS_Model20252602" / "cascade_strategy.json")        
        model_names = self.loadmodelnames(Path(__file__).parent.parent.parent  / "strategy" / "AGNEWS_Model20252602/")
        self.scorer = dict()
        for name in model_names:
            path1 = (Path(__file__).parent.parent.parent  / "strategy" / "AGNEWS_Model20252602/" / name / "")
            self.MyScores[name]=Score()
            self.MyScores[name].load(path1)
            self.scorer[name]  = self.MyScores[name].get_model()
            #logging.critical(f"Loaded scorer for service: {name}")
        #logging.critical(f"Loaded model names: {model_names}")
        #logging.critical(f"Available scorers: {self.MyScores.keys()}")
        return
    
    def loadmodelnamesold(self,loadpath):
        directories = []
        for entry in os.scandir(loadpath):
            if entry.is_dir():
                for sub_entry in os.scandir(entry.path):
                    if sub_entry.is_dir():
                        subdirectory_name = os.path.relpath(sub_entry.path, loadpath)
                        directories.append(subdirectory_name)
        keys = directories
        return keys

    def loadmodelnames(self, loadpath):
      directories = []
      for dirpath, dirnames, _ in os.walk(loadpath):
          if not dirnames:  # If there are no more subdirectories, it's the last directory
            subdirectory_name = os.path.relpath(dirpath, loadpath)
            directories.append(subdirectory_name)
      keys = directories
      return keys  # Return the directories list if needed


    def save(self, savepath="strategy/HEADLINES/"):
        # Save both Scores and LLChains to the disk
        if not os.path.exists(savepath):
            os.makedirs(savepath)
        strategy_path = savepath+"cascade_strategy.json"    
        self.LLMChain.savestrategy(strategy_path)
        if(self.no_scorer_train):
            return
        for name in self.MyScores:
            path1 = savepath+name+"/"
            self.MyScores[name].save(path1)
        return
    
    def train(self,
              trainingdata=None,
              budget=0.1,
              cascade_depth=3,
              service_names =['openaichat/gpt-3.5-turbo','openaichat/gpt-4'],
              metric="em",
              genparams=GenerationParameter(max_tokens=50, temperature=0.1, stop=['\n']),
              no_scorer_train=False,
              score_type='DistilBert',
              score_test_size=0.55,
              prefix="",
              ):
        self.no_scorer_train = no_scorer_train
        self.score_type = score_type
        self.score_test_size = score_test_size
        self.prefix = prefix
        # Three major steps
        # Step 1: evaluate all services on the given dataset
        train, test = train_test_split(trainingdata, test_size=0.01)
        print("train and test size",len(train),len(test))
        model_perf_train = self.evaluateall(train,service_names=service_names,metric=metric,genparams=genparams)
        model_perf_test = self.evaluateall(test,service_names=service_names,metric=metric,genparams=genparams)
        # Step 2: Build the scorer
        if(no_scorer_train):
            print("directly get the scorers")
            scorers = self.get_scorers()
            print("")
        else:
            scorers = self.build_scorers(model_perf_train)
        # Step 3: Build the cascade
        #self.build_cascade(model_perf_test, scorers = scorers, budget=budget, cascade_depth=cascade_depth,metric=metric)
        self.build_cascade(model_perf_train, scorers = scorers, budget=budget, cascade_depth=cascade_depth,metric=metric)
        return model_perf_test
    
    def get_completion(self, query, genparams):
        # Check cache first if enabled
        if self.use_cache:
            cached_response, cached_model = self.cache.get_from_cache(query)
            if cached_response is not None:
                self.cost = 0  # Cache hits cost nothing
                return cached_response, cached_model

        # Original completion logic
        LLMChain = self.LLMChain
        MyLLMEngine = self.MyLLMEngine
        cost = 0 
        LLMChain.reset()
        prefix = self.prefix
        res = None
        model_used = None
        
        while(1):
            service_name, score_thres = LLMChain.nextAPIandScore()
            if(service_name==None):
                break
            logging.critical(f"Using service: {service_name}")
            res = MyLLMEngine.get_completion(query=query,service_name=service_name,genparams=genparams)
            cost += MyLLMEngine.get_cost()
            t1 = query+" "+res
            t2 = t1.removeprefix(prefix)
            service_name = service_name.replace("/", "\\")
            score = self.MyScores[service_name].get_score(scorer_text(t2))
            if(self.score_noise_injection==True):
                score+=random.random()*self.eps
            if(score>1-score_thres):
                model_used = service_name
                # Add successful response to cache
                if self.use_cache:
                    self.cache.add_to_cache(query, res, model_used)
                break
        self.cost = cost
        return res if res is not None else "", model_used

    def get_completion_batch(self, queries, genparams):
        result = list()
        for query in queries:
            ans1, model_used = self.get_completion(query=query[0], genparams=genparams)
            cost = self.get_cost()
            result.append({'_id':query[2],'answer':ans1,'ref_answer':query[1],'cost':cost, 'model_used': model_used})
        result = pandas.DataFrame(result)
        return result
        
    def get_cost(self):
        return self.cost

    def _get_response(self,
                      data,
                      genparams,
                      service_name,
                      ):
        # data is a list
        # data[i][0]: query, data[i][1]: answer
        result = list()
        MyLLMEngine = self.MyLLMEngine
        for i in range(len(data)):
            query = data[i][0]
            temp = dict()
            temp['true_answer']= data[i][1] 
            temp['_id'] = data[i][2]
            temp['query'] = query
            temp['answer'] = MyLLMEngine.get_completion(query=query,service_name=service_name,genparams=genparams)
            temp['latency'] = MyLLMEngine.get_latency()
            temp['cost'] = MyLLMEngine.get_cost()
            result.append(temp)
        return result
 
    def  build_scorers(self,model_perf_train):
        self.scorer = dict()
        for name in model_perf_train:
            self.MyScores[name], self.scorer[name] = self._build_scorer(model_perf_train[name])
        return self.scorer

    def get_scorers(self):
        return self.scorer

    def _build_scorer(self,res_and_eval):
        #train, test = train_test_split(res_and_eval, test_size=0.2)
        #print("res_and_eval",res_and_eval)
        #traintext = list((res_and_eval['query']+res_and_eval['answer']).apply(scorer_text))
        prefix = self.prefix
        #traintext = list((res_and_eval['query'] + res_and_eval['answer']).apply(lambda x: scorer_text(x).removeprefix(prefix)))
        traintext = list((res_and_eval['query'] +" "+ res_and_eval['answer']).apply(lambda x: scorer_text(x.removeprefix(prefix))))



        trainlabel = list(res_and_eval['quality'])
        MyScore = Score(score_type=self.score_type, test_size=self.score_test_size)
        model = MyScore.train(traintext,trainlabel)
        return MyScore, model
    
    def get_scores(self, data, name):
        eps=self.eps
        #model = self.scorer[name]
        scores_dict = dict()
        rawdata = data[['_id','query','answer']].to_dict(orient='records')
        for ptr in rawdata:
            text0 = ptr['query']+" "+ptr['answer']
            text0 = text0.removeprefix(self.prefix)
            text = scorer_text(text0)
            score1 = self.MyScores[name].get_score(text)
            if(self.score_noise_injection==True):
              score1+=random.random()*eps
            scores_dict[ptr['_id']] = score1
        return scores_dict

    def evaluateall(self,train,service_names,metric,genparams):
        api_responses = dict()
        for name in service_names:
            # step 1: get the answers from all API
            response = self._get_response(data=train, genparams=genparams,service_name=name)
            # step 2: evaluate the performance
            res_and_eval = self._evaluate(response, metric=metric)
            api_responses[name] = res_and_eval
        return api_responses

    def _evaluate(self,response, metric='em'):
        for i in range(len(response)):
            ptr = response[i]
            score = evaluate(prediction = ptr['answer'], ground_truth=ptr['true_answer'], metric=metric)
            response[i]['quality'] = score
        result = pandas.DataFrame(response)
        return result
    
    def build_cascade(self,model_perf_test, scorers, budget, cascade_depth,metric):
        LLMChain1 = LLMChain(metric=metric,L_max=cascade_depth)
        LLMChain1.setbudget(budget=budget)
        responses = dict()
        scores = dict()
        if(self.batch_build):
            try:
                responses = self.responses
                labels = self.labels
                scores = self.scores         
                print("scores",scores)

                LLMChain1.train(responses,labels,scores)
                self.LLMChain = LLMChain1
                return

            except:
                print("first train")

        for key in model_perf_test:
            labels = model_perf_test[key][['_id','true_answer']].rename(columns={'true_answer': 'answer'}).to_dict(orient='records')
            responses[key] = table2json(model_perf_test[key])
            scores[key] = self.get_scores(model_perf_test[key],name=key)
            tempsave(labels,responses[key],scores[key],key)
        #print("responses",responses)  
        #print("labels",labels) 
        #print("scores",scores)
        self.responses = responses
        self.labels = labels
        self.scores = scores         
        LLMChain1.train(responses,labels,scores)
        self.LLMChain = LLMChain1
        return
    
def table2json(df):
    # Convert DataFrame to the desired dictionary format
    result_dict = {}
    for _, row in df.iterrows():
        answer = row['answer']
        cost = row['cost']
        _id = row['_id']
        quality = row['quality']
    
        # Update "answer" key
        if 'answer' not in result_dict:
            result_dict['answer'] = dict()
        result_dict['answer'][_id]= answer
    
        # Update "cost" key
        if 'cost' not in result_dict:
            result_dict['cost'] = dict()
        result_dict['cost'][_id]= cost
    
        # Update "quality" key
        if 'quality' not in result_dict:
            result_dict['quality'] = dict()
        result_dict['quality'][_id]= quality

    result_dict['sp'] = dict()
    result_dict['logprobs'] = dict()
    return result_dict

class strategy():
    def __init__(self):
        return