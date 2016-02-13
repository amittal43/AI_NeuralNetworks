from NeuralNetUtil import buildExamplesFromCarData,buildExamplesFromPenData
from NeuralNet import buildNeuralNet
import cPickle 
from math import pow, sqrt
# from __main__ import name

def average(argList):
    return sum(argList)/float(len(argList))

def stDeviation(argList):
    mean = average(argList)
    diffSq = [pow((val-mean),2) for val in argList]
    return sqrt(sum(diffSq)/len(argList))


penData = buildExamplesFromPenData() 
def testPenData(hiddenLayers = [24]):
    return buildNeuralNet(penData,maxItr = 200, hiddenLayerList =  hiddenLayers)

carData = buildExamplesFromCarData()


def testCarData(hiddenLayers = [16]):
    return buildNeuralNet(carData,maxItr = 200,hiddenLayerList =  hiddenLayers)


def main():
    
    car = []
    pen = []
    for i in range(0,5):
        x,y = testCarData()
        a,b = testPenData()
        car.append(y)
        pen.append(b)
         
     
    print 'Car Data \n'
#     print ' Max %d '%(max(car))
    print car
    print ' Average %f '%(average(car))
    print ' Standard Deviation %f '%(stDeviation(car))
     
    print 'Pen Data \n'
    print pen
#     print ' Max %d ' %(max(pen))
    print ' Average %f ' %(average(pen))
    print ' Standard Deviation %f ' %(stDeviation(pen))

    
        
    for j in range(0,41,5):
        car2 = []
        pen2 = []
        for i in range(0,5):
            x,y = testCarData([j])
            a,b = testPenData([j])
            car2.append(y)
            pen2.append(b)
      
        print "Car Data \n"
        print car2
#         print " Max %d\n" %(max(car2))
        print " Average %f \n" %(average(car2))
        print " Standard Deviation %f \n" %(stDeviation(car2))
         
        print "Pen Data \n"
        print pen2
#         print " Max %d \n" %(max(pen2))
        print " Average %f \n" %(average(pen2))
        print " Standard Deviation %f \n" %(stDeviation(pen2))
   
   
    
#     for j in range(10,21,5):
#         car2 = []
#         pen2 = []
#         for i in range(0,5):
#             x,y = testCarData([j])
#             a,b = testPenData([j])
#             car2.append(y)
#             pen2.append(b)
#       
#         print "Car Data \n"
#         print " Max %d\n " %(max(car2))
#         print " Average %d\n " %(average(car2))
#         print " Standard Deviation %d\n " %(stDeviation(car2))
#          
#         print "Pen Data"
# #         print " Max %d \n" %(max(pen2))
#         print " Average %d \n" %(average(pen2))
#         print " Standard Deviation %d \n" %(stDeviation(pen2))
#          
#      
#     for j in range(20,41,5):
#         car2 = []
#         pen2 = []
#         for i in range(0,5):
#             x,y = testCarData([j])
#             a,b = testPenData([j])
#             car2.append(y)
#             pen2.append(b)
#       
# #         print " Max %d " %(max(car2))
#         print " Average %d " %(average(car2))
#         print " Standard Deviation %d " %(stDeviation(car2))
#          
#         print "Pen Data"
#         print " Max %d " %(max(pen2))
#         print " Average %d " %(average(pen2))
#         print " Standard Deviation %d " %(stDeviation(pen2))
    
        
if __name__ == '__main__':
    main() 
    
    
