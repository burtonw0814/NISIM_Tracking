import os

def init_tmp_dirs(model_id):

    #####################################################
    # try to make directory for storing models
    try:
        os.mkdir("./models/")
    except:
        []
    #####################################################
    
    #####################################################
    # Create directories for caching training batches, helps with batch prep latency
    
    # Clear temp batch files
    print("Clearing temp batches files")
    bx_dir="./tmp/tmp_bx_"+model_id+"/"
    by_dir="./tmp/tmp_by_"+model_id+"/"
    s_idx_dir="./tmp/tmp_s_idx_"+model_id+"/"

    # try to makedir
    try:
        os.mkdir("./tmp/")
    except:
        []
    try:
        os.mkdir(bx_dir)
    except:
        []
    try:
        os.mkdir(by_dir)
    except:
        []
    try:
        os.mkdir(s_idx_dir)
    except:
        []

    my_files=os.listdir(bx_dir)
    for i in my_files:
        os.remove(bx_dir+"/"+i)
    my_files=os.listdir(by_dir)
    for i in my_files:
        os.remove(by_dir+"/"+i)
    my_files=os.listdir(s_idx_dir)
    for i in my_files:
        os.remove(s_idx_dir+"/"+i)
    print("Done")
    #####################################################
    
    return
           
                    
                    
