package ml.classifiers;
import ml.data.Example;
import ml.data.CrossValidationSet;
import ml.data.DataSet;
import ml.data.DataSetSplit;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.concurrent.atomic.DoubleAdder;

/**
 * @author Pete Boyle and Luca Snoey
 * Assignment 7b
 */

public class Experiments {

    /**
     * Creates a cross validation and classifies using a classifier factory of choice
     * Averages values across iterations
     * @param args
     */
    public static void main(String[] args){

       // NBClassifier classifier = new NBClassifier();

        DataSet data = new DataSet("data/wines.train", 1);

        //classifier.train(data);

        //System.out.println(classifier.classify(data.getData().get(0)));

        
        CrossValidationSet crossSet = new CrossValidationSet(data, 10);

        DataSetSplit split = crossSet.getValidationSet(0);

        DataSet train = split.getTrain();
               
        DataSet test = split.getTest();

        //for(double i = 0; i<10; i++){
            //double accuracy = 0;
            
            NBClassifier classifier = new NBClassifier();

            //a.setLambda(i/100);

            classifier.train(train);

            ArrayList<Example> testSet = test.getData();

            
            

              // Sort the List by the second double value
            

            ArrayList<ArrayList<Double>> list = new ArrayList<ArrayList<Double>>();
            //System.out.println("Size:" + testSet.size());
            /*for (int h = 0; h < testSet.size(); h++){

                Double c = classifier.classify(testSet.get(h));
                Double confidence = classifier.confidence(testSet.get(h));

                ArrayList<Double> temp = new ArrayList<Double>();
                temp.add(c);
                temp.add(confidence);

                list.add(temp);

                //System.out.println("Classification: " + c + "Confidence:" + a.confidence(testSet.get(h)))
            }*/

           /*  Collections.sort(list, new Comparator<ArrayList<Double>>() {
                @Override
                public int compare(ArrayList<Double> a, ArrayList<Double> b) {
                    return b.get(1).compareTo(a.get(1));
                }
            });*/

            Collections.sort(testSet, new Comparator<Example>() {
                @Override
                public int compare(Example a, Example b) {
                    double confidenceA = classifier.confidence(a);
                    double confidenceB = classifier.confidence(b);
                    
                    if (confidenceA > confidenceB) {
                        return -1;
                    } else if (confidenceA < confidenceB) {
                        return 1;
                    } else {
                        return 0;
                    }
                }
            });

            /* 
            //double accuracy = 0;
            double testCount = 0;
            for(int i = 0; i<testSet.size(); i++){
                if (list.get(i).get(0) == testSet.get(i).getLabel()){
                    testCount ++;
                }
                System.out.println(testCount/i);
                // + "Confidence: " + list.get(i).get(1));
            }*/

            double testCount = 0;
            double counter = 1;
            for(Example e : testSet){
                System.out.println(classifier.confidence(e));
            }
            System.out.println("ACCURACY");
            for(Example e : testSet){
                if(e.getLabel() == classifier.classify(e)){
                    testCount ++;
                }

                System.out.println(testCount/counter);
               // System.out.println(classifier.confidence(e));
                counter ++;
            }

        //}   

        /* 
       
        //goes through each cross set and accesses the train and test data. Trains and classifies, calculating accuracy.
        for(int i = 0; i < crossSet.getNumSplits(); i++){

            double accuracy = 0;

            DataSetSplit split = crossSet.getValidationSet(i);

            DataSet train = split.getTrain();
               
            DataSet test = split.getTest();

            NBClassifier a = new NBClassifier();

            a.train(train);

            ArrayList<Example> testSet = test.getData();

            double testCount = 0;
            //System.out.println("Size:" + testSet.size());
            for (int h = 0; h < testSet.size(); h++){

                Double c = a.classify(testSet.get(h));
                //System.out.println("Classification: " + c + "Confidence:" + a.confidence(testSet.get(h)));

                if (c == testSet.get(h).getLabel()){

                    testCount ++;

                }

            }

            accuracy = testCount / testSet.size();
            System.out.println("Accuracy: " + accuracy);
            

        }

        
        */
    } 
}