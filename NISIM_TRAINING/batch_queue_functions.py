import multiprocessing as mp
import numpy as np

# Custom class that processes training batches in parallel using Python built-in multiprocessing library

class async_batch_queue():
    
    def __init__(self, num_inst, TOT_LIST, batch_method):
    
        self.num_inst=num_inst
        self.inst_list=[None]*self.num_inst;
        self.live_list=np.zeros((self.num_inst));
        self.batch_method=batch_method
        
        print('GENERATING INITIAL INSTANCES')
        
        #self.instance_queue=[self.batch_method(TOT_LIST) for x in range(self.num_inst)]
        self.pool = mp.Pool(processes=2)
        self.instance_queue=[self.pool.apply(self.batch_method, args=(TOT_LIST,)) for x in range(self.num_inst)]
        self.pool.close()
        
        print('INITIAL INSTANCES GENERATED')
        self.pool2 = mp.Pool(processes=2) # Initialize separate pool for creating batches during training
        
        return
        
    def apply_batches(self, TOT_LIST):

        for ll in range(self.num_inst):
            if self.live_list[ll]==0: # Check if an instance is currently being updated
                try_flag=1; try_ct=0;
                while try_flag==1 and try_ct<5:                   
                    if True: #try:
                        
                        # Set different rando seed
                        TOT_LIST=list(TOT_LIST)
                        TOT_LIST[-1]=np.random.randint(low=0, high=1000000)
                        TOT_LIST=tuple(TOT_LIST)
                    
                        self.inst_list[ll]=self.pool2.apply_async(self.batch_method, args=(TOT_LIST,)) # Process new batch
                        self.live_list[ll]=1; try_flag=0;
                        try_flag=0; try_ct+=1;
        return
    
    def retrieve_batches(self):

        for ll in range(self.num_inst): 
            if self.live_list[ll]==1:
                if self.inst_list[ll].ready(): # If ready, retrieve new instance
                    try_flag=1; try_ct=0;
                    while try_flag==1 and try_ct<5:
                        if True: #try: 
                            self.instance_queue[ll]=self.inst_list[ll].get()
                            self.inst_list[ll]=[]; self.live_list[ll]=0
                            try_flag=0; try_ct+=1
        return

    def sample_batch(self):
        inst=np.random.randint(low=0, high=self.num_inst) 
        batch=self.instance_queue[inst]
        return batch
        
    def close_batch_queue(self):
        self.pool2.close()
        return
        
        
        
        
        
