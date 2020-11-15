
import subprocess
from pathlib import Path

def minor_output_evd_cr(f,instance,index):
    if instance['time_period']=="busy":
        f.write("Time(Busy,"+index+")\n")
        f.write("!Time(Normal,"+index+")\n")
    elif instance['time_period']=="normal":
        f.write("Time(Normal,"+index+")\n")
        f.write("!Time(Busy,"+index+")\n")           
            
    if instance['perception']=="crowded":
        f.write("Perception(Crowded,"+index+")\n")
        f.write("!Perception(Empty,"+index+")\n")          
    elif instance['perception']=="empty":
        f.write("!Perception(Crowded,"+index+")\n")
        f.write("Perception(Empty,"+index+")\n")          
    f.write("\n")

def minor_output_evd_r_cr(f,instance,index):
    if instance['time_period']=="busy":
        f.write("Time(Busy,"+index+")\n")
        f.write("!Time(Normal,"+index+")\n")
    elif instance['time_period']=="normal":
        f.write("Time(Normal,"+index+")\n")
        f.write("!Time(Busy,"+index+")\n")           
            
    f.write("!Perception(Crowded,"+index+")\n")
    f.write("Perception(Empty,"+index+")\n")          
    f.write("\n")

def minor_output_evd(f,instance,index):
    if instance['weather']=="sunny":
        f.write("Weather(Sunny,"+index+")\n")
        f.write("!Weather(Rainy,"+index+")\n")
    elif instance['weather']=="rainy":
        f.write("!Weather(Sunny,"+index+")\n")
        f.write("Weather(Rainy,"+index+")\n")
    if instance['time_period']=="busy":
        f.write("Time(Busy,"+index+")\n")
        f.write("!Time(Normal,"+index+")\n")
    elif instance['time_period']=="normal":
        f.write("Time(Normal,"+index+")\n")
        f.write("!Time(Busy,"+index+")\n")
    if instance['perception']=="crowded":
        f.write("Perception(Crowded,"+index+")\n")
        f.write("!Perception(Empty,"+index+")\n")
    elif instance['perception']=="empty":
        f.write("!Perception(Crowded,"+index+")\n")
        f.write("Perception(Empty,"+index+")\n")          
    f.write("\n")

def minor_output_data(f,instance,index):
    if instance['weather']=="sunny":
        f.write("Weather(Sunny,"+index+")\n")
        f.write("!Weather(Rainy,"+index+")\n")
    elif instance['weather']=="rainy":
        f.write("!Weather(Sunny,"+index+")\n")
        f.write("Weather(Rainy,"+index+")\n")
      
    if instance['time_period']=="busy":
        f.write("Time(Busy,"+index+")\n")
        f.write("!Time(Normal,"+index+")\n")
    elif instance['time_period']=="normal":
        f.write("Time(Normal,"+index+")\n")
        f.write("!Time(Busy,"+index+")\n")
                        
    if instance['cr']=="crowded":
        f.write("Road(Crowded,"+index+")\n")
        f.write("!Road(Empty,"+index+")\n")
            
    elif instance['cr']=="empty":
        f.write("!Road(Crowded,"+index+")\n")
        f.write("Road(Empty,"+index+")\n")
                     
    if instance['perception']=="crowded":
        f.write("Perception(Crowded,"+index+")\n")
        f.write("!Perception(Empty,"+index+")\n")
            
    elif instance['perception']=="empty":
        f.write("!Perception(Crowded,"+index+")\n")
        f.write("Perception(Empty,"+index+")\n")
            
    if instance['co']=="cooperative":
        f.write("Cooperative("+index+")\n")
        #f.write("Cooperative("+index+")\n")
    elif instance['co']=="not cooperative": 
        f.write("!Cooperative("+index+")\n")
    f.write("\n")

def minor_output_data_r(f,instance,index):
    if instance['weather']=="sunny":
        f.write("Weather(Sunny,"+index+")\n")
        f.write("!Weather(Rainy,"+index+")\n")
    elif instance['weather']=="rainy":
        f.write("!Weather(Sunny,"+index+")\n")
        f.write("Weather(Rainy,"+index+")\n")
      
    if instance['time_period']=="busy":
        f.write("Time(Busy,"+index+")\n")
        f.write("!Time(Normal,"+index+")\n")
    elif instance['time_period']=="normal":
        f.write("Time(Normal,"+index+")\n")
        f.write("!Time(Busy,"+index+")\n")
                        
    if instance['cr']=="crowded":
        f.write("Road(Crowded,"+index+")\n")
        f.write("!Road(Empty,"+index+")\n")
            
    elif instance['cr']=="empty":
        f.write("!Road(Crowded,"+index+")\n")
        f.write("Road(Empty,"+index+")\n")
            
    f.write("!Perception(Crowded,"+index+")\n")
    f.write("Perception(Empty,"+index+")\n")
            
    if instance['co']=="cooperative":
        f.write("Cooperative("+index+")\n")
    elif instance['co']=="not cooperative": 
        f.write("!Cooperative("+index+")\n")
    f.write("\n")

def minor_output_evd_r(f,instance,index):           
    if instance['weather']=="sunny":
        f.write("Weather(Sunny,"+index+")\n")
        f.write("!Weather(Rainy,"+index+")\n")
    elif instance['weather']=="rainy":
        f.write("!Weather(Sunny,"+index+")\n")
        f.write("Weather(Rainy,"+index+")\n")
      
    if instance['time_period']=="busy":
        f.write("Time(Busy,"+index+")\n")
        f.write("!Time(Normal,"+index+")\n")
    elif instance['time_period']=="normal":
        f.write("Time(Normal,"+index+")\n")
        f.write("!Time(Busy,"+index+")\n")
            
    f.write("!Perception(Crowded,"+index+")\n")
    f.write("Perception(Empty,"+index+")\n")     
    f.write("\n")

def output_evidence(f,instance,index):
    #f=open(f)    
    if instance['weather']=="sunny":
          f.write("Weather(Sunny,"+index+")\n")
          f.write("!Weather(Rainy,"+index+")\n")
    elif instance['weather']=="rainy":
        f.write("!Weather(Sunny,"+index+")\n")
        f.write("Weather(Rainy,"+index+")\n")
    
    if instance['time_period']=="busy":
        f.write("Time(Busy,"+index+")\n")
        f.write("!Time(Normal,"+index+")\n")
    elif instance['time_period']=="normal":
        f.write("Time(Normal,"+index+")\n")
        f.write("!Time(Busy,"+index+")\n")
      
    if instance['vehicle_type']=="sedan":
        f.write("Vehicle(Car,"+index+")\n")
        f.write("!Vehicle(Sport,"+index+")\n")
        f.write("!Vehicle(Truck,"+index+")\n")
      
    elif instance['vehicle_type']=="sport":
        f.write("Vehicle(Sport,"+index+")\n")
        f.write("!Vehicle(Car,"+index+")\n")
        f.write("!Vehicle(Truck,"+index+")\n")
               
    elif instance['vehicle_type']=="truck":
        f.write("Vehicle(Truck,"+index+")\n")
        f.write("!Vehicle(Car,"+index+")\n")
        f.write("!Vehicle(Sport,"+index+")\n")
            
    if instance['perception']=="crowded":
        f.write("Perception(Crowded,"+index+")\n")
        f.write("!Perception(Empty,"+index+")\n")
            
    elif instance['perception']=="empty":
        f.write("!Perception(Crowded,"+index+")\n")
        f.write("Perception(Empty,"+index+")\n")
            
    f.write("\n")

def output_data(data_file,instance,index):
    #f=open(f)    
    f=open("reasoner/"+data_file,"a")
    if instance['weather']=="sunny":
        f.write("Weather(Sunny,"+index+")\n")
        f.write("!Weather(Rainy,"+index+")\n")
    elif instance['weather']=="rainy":
        f.write("!Weather(Sunny,"+index+")\n")
        f.write("Weather(Rainy,"+index+")\n")
      
    if instance['time_period']=="busy":
        f.write("Time(Busy,"+index+")\n")
        f.write("!Time(Normal,"+index+")\n")
    elif instance['time_period']=="normal":
        f.write("Time(Normal,"+index+")\n")
        f.write("!Time(Busy,"+index+")\n")
      
    if instance['vehicle_type']=="sedan":
        f.write("Vehicle(Car,"+index+")\n")
        f.write("!Vehicle(Sport,"+index+")\n")
        f.write("!Vehicle(Truck,"+index+")\n")
      
    elif instance['vehicle_type']=="sport":
        f.write("Vehicle(Sport,"+index+")\n")
        f.write("!Vehicle(Car,"+index+")\n")
        f.write("!Vehicle(Truck,"+index+")\n")
               
    elif instance['vehicle_type']=="truck":
        f.write("Vehicle(Truck,"+index+")\n")
        f.write("!Vehicle(Car,"+index+")\n")
        f.write("!Vehicle(Sport,"+index+")\n")
            
    if instance['cr']=="crowded":
        f.write("Road(Crowded,"+index+")\n")
        f.write("!Road(Empty,"+index+")\n")
            
    elif instance['cr']=="empty":
        f.write("!Road(Crowded,"+index+")\n")
        f.write("Road(Empty,"+index+")\n")
            
    if instance['perception']=="crowded":
        f.write("Perception(Crowded,"+index+")\n")
        f.write("!Perception(Empty,"+index+")\n")
            
    elif instance['perception']=="empty":
        f.write("!Perception(Crowded,"+index+")\n")
        f.write("Perception(Empty,"+index+")\n")
                    
    if instance['co']=="cooperative":
        f.write("Cooperative("+index+")\n")
    elif instance['co']=="not cooperative": 
        f.write("!Cooperative("+index+")\n")
        f.write("\n")
    f.close()

def learn_weights(input_file="autocar.mln",output_file="trained.mln",train_data="train.db"):
    infer_path = Path.cwd()/'reasoner/learnwts'
    assert Path(infer_path).is_file(), 'learnwts path does not exist'
    #subprocess.check_output([infer_path,"-i",mln_file,"-r",result_file,"-e",evidence_file, "-q", query])
    #why this does not work, but in parser pomdp it worked
    subprocess.run(["./learnwts","-g","-i",input_file,"-o",output_file,"-t",train_data],cwd='reasoner')    

def infer(mln_file="trained.mln",result_file="query.result",evidence_file="query.db",query="Cooperative"):
      
    infer_path = Path.cwd()/'reasoner/infer'
    assert Path(infer_path).is_file(), 'infer path does not exist'
    #subprocess.check_output([infer_path,"-i",mln_file,"-r",result_file,"-e",evidence_file, "-q", query])
    #why this does not work, but in parser pomdp it worked
    subprocess.run(["./infer","-i",mln_file,"-r",result_file,"-e",evidence_file, "-q", query,"-maxSteps", "200"],cwd='reasoner')

def infer_cr(mln_file="trained.mln",result_file="query_cr.result",evidence_file="query.db",query="Road"):    
    infer_path = Path.cwd()/'reasoner/infer'
    assert Path(infer_path).is_file(), 'infer path does not exist'
    #subprocess.check_output([infer_path,"-i",mln_file,"-r",result_file,"-e",evidence_file, "-q", query])
    #why this does not work, but in parser pomdp it worked
    subprocess.run(["./infer","-i",mln_file,"-r",result_file,"-e",evidence_file, "-q", query,"-maxSteps", "200"],cwd='reasoner')     

def read_result(file_name):
    f=open("reasoner/"+file_name,"r")
    co_list=[]
    for line in f:
        predict=line.split()[-1]
        co=round(float(predict),4)
        co_list.append(co)
    f.close()
    return co_list
       
def read_result_cr(file_name):
    f=open("reasoner/"+file_name,"r")
    co_list=[]
    for line in f:
        if line.find("Empty")!=-1:
            predict=line.split()[-1]
            co=round(float(predict),4)
            co_list.append(co)
    f.close()
    return co_list
