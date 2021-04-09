#Agradecimientos a Sergio y Ángel por la base de datos y esta parte del código.

import os
import pickle
import numpy.random as rnd

def loadData(DatabaseName):
    
    print("Loading database...")
    
    datapath=DatabaseName
    
    #Creating training data: #####################################
    trainingData=[]
    
    experiments=[]; solutions=[] #Subarrays which will go into trainingData
    
    
    pathExp=os.path.join(datapath,"Experiments")
    pathExpSol=os.path.join(datapath,"SolutionsExp")
        
    a=1
    fileExist=True
    while(fileExist):
        experiment="experiment_"+str(a)+".pickle"
        try:
            file=open(os.path.join(pathExp,experiment),"rb")
            fromfile=pickle.load(file)
            file.close()
            experiments.append(fromfile)
            a+=1
        except:
            fileExist=False           
            print("...")
            
    a=1        
    fileExist=True
    while(fileExist):
        solution="solution_"+str(a)+".pickle"
        try:
            file=open(os.path.join(pathExpSol,solution),"rb")
            fromfile=pickle.load(file)
            file.close()
            solutions.append(fromfile)
            a+=1
        except:
            fileExist=False
            print("...")
        
    for exp, sol in zip(experiments, solutions):
        trainingData.append([exp, sol])
    print("...")
    
    #Creating test data: #####################################
    testData=[]
    
    testexperiments=[]; testsolutions=[] #Subarrays which will go into trainingData
        
    pathTest=os.path.join(datapath,"Test")
    pathTestSol=os.path.join(datapath,"SolutionsTest")
  
    a=1
    fileExist=True
    while(fileExist):
        test="test_"+str(a)+".pickle"
        try:
            file=open(os.path.join(pathTest,test),"rb")
            fromfile=pickle.load(file)
            file.close()
            testexperiments.append(fromfile)
            a+=1
        except:
            fileExist=False           
            print("...")
            
    a=1        
    fileExist=True
    while(fileExist):
        solution="solution_"+str(a)+".pickle"
        try:
            file=open(os.path.join(pathTestSol,solution),"rb")
            fromfile=pickle.load(file)
            file.close()
            testsolutions.append(fromfile)
            a+=1
        except:
            fileExist=False
            print("...")
        
    for exp, sol in zip(testexperiments, testsolutions):
        testData.append([exp, sol])
    print("...")
    
    
    #Creating validation data: #####################################
    validationData=[]
    
    val_experiments=[]; val_solutions=[] #Subarrays which will go into trainingData
    
    
    pathValExp=os.path.join(datapath,"Spare")
    pathValSol=os.path.join(datapath,"SolutionsSpare")

    a=1
    fileExist=True
    while(fileExist):
        validation="validation_"+str(a)+".pickle"
        try:
            file=open(os.path.join(pathValExp,validation),"rb")
            fromfile=pickle.load(file)
            file.close()
            val_experiments.append(fromfile)
            a+=1
        except:
            fileExist=False           
            print("...")
            
    a=1        
    fileExist=True
    while(fileExist):
        solution="solution_"+str(a)+".pickle"
        try:
            file=open(os.path.join(pathValSol,solution),"rb")
            fromfile=pickle.load(file)
            file.close()
            val_solutions.append(fromfile)
            a+=1
        except:
            fileExist=False
            print("...")
        
    for exp, sol in zip(val_experiments, val_solutions):
        validationData.append([exp, sol])
    print("...")
    
    
    #Shuffling the arrays:
    rnd.shuffle(trainingData)
    rnd.shuffle(testData)
    rnd.shuffle(validationData)
    print("...")
    print("Data loaded.")
    
        
    return [trainingData, testData, validationData]