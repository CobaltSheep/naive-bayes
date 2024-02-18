package ml.classifiers;

import java.util.ArrayList;
import java.util.HashMap;

import ml.data.DataSet;
import ml.data.Example;
import ml.utils.HashMapCounter;

/**
 * @author Pete Boyle and Luca Snoey
 * Assignment 7b
 */

/**
 * Naive-Bayes Classifier that predicts a label based on
 * the log probability of an example and label pairing existing in the
 * dataset.
 */
public class NBClassifier implements Classifier{

    private double lambda = 0.01;

    private boolean onlyPos = true;

    //hashmap which stores labels as keys and hashmaps for each feature/count as values
    private HashMap<Double, HashMapCounter<Integer>> counts;
    //hashmap that stores labels as keys and how often the label appears as values
    private HashMapCounter<Double> labelCounts;
    //stores the size of the dataset
    private int dSetSize = 0;
    //stores the dataset as attribute
    private DataSet global_data;
    
    /**
     * Zero parameter constructor
     */
    public NBClassifier(){

    }

    /**
     * Sets the lambda value for regularization
     * @param lam
     */
    public void setLambda(double lam){
        this.lambda = lam;
    }

    /**
     * Sets if only positive features will be used for
     * probability calculations
     * @param b
     */
    public void setUseOnlyPositiveFeatures(boolean b){
        if(b){
            this.onlyPos = true;
        }
        else{
            this.onlyPos = false;
        }
    }

    /**
     * Trains the classifier on a given dataset.
     * Stores counts of each instance of a feature for each label.
     * @param data
     */
    @Override
    public void train(DataSet data) {

        //stores dataset as attribute
        global_data = data;
       
        
        ArrayList<Example> examples = data.getData();

        dSetSize = examples.size();

        //clears hashmaps
        counts = new HashMap<Double, HashMapCounter<Integer>>();
        labelCounts = new HashMapCounter<Double>();

            for(Example e : examples){

                //creates a hashmap for a new label
                if(!counts.containsKey(e.getLabel())){
                    counts.put(e.getLabel(), new HashMapCounter<Integer>());
                }   
                //increments the count for a label
                labelCounts.increment(e.getLabel(), 1);

                for(int featureIndex: e.getFeatureSet()){

                    if(e.getFeature(featureIndex) > 0){
                        //increments count for a feature given a label
                        counts.get(e.getLabel()).increment(featureIndex, 1);
                    }
                }


            }
        
    }  

    /**
     * Calculates the probability of an example with a given label
     * Goes through every feature in the featureset and adds the log of its calculated probability
     * Sums these probabilities and returns the sum
     * Only includes "positive" features
     * @param ex
     * @param label
     * @return
     */
    private double getLogProbPos(Example ex, double label){

        // initialize our final product with the probability of the label
        double sum = Math.log10((double) labelCounts.get(label)/dSetSize);
       
        // loop through the features in the example
         for(int feature : ex.getFeatureSet()){
            
            // calculate total prob using log function 
            if(ex.getFeature(feature) > 0){
                sum += Math.log10(getFeatureProb(feature, label));
            }
           
        }

        return sum;


    }

    /**
     * Calculates the probability of an example with a given label
     * Goes through every feature in the dataset and adds the log of its calculated probability
     * Adds the probabiltiy if the feature is positive, and if negative adds 1-probability
     * Sums the logs and returns the sum
     * @param ex
     * @param label
     * @return
     */
    private double getLogProbAll(Example ex, double label){

        // initialize our final product with the probability of the label
        double sum = Math.log10((double)labelCounts.get(label)/dSetSize);
        //System.out.println("INITSUM: " + sum);
        // loop through all features
        for(int feature : global_data.getAllFeatureIndices()){

            // if the feature is in our example, calculate the probability
            if(ex.getFeature(feature) > 0){
                //System.out.println("FeatureProb: " + Math.log10(getFeatureProb(feature, label)));
                sum += Math.log10(getFeatureProb(feature, label));
               

            }
            // if feature is not in example, do 1 - prob 
            else{
                //System.out.println("FeatureProb: " + Math.log10(getFeatureProb(feature, label)));
                if(getFeatureProb(feature, label) != Double.NEGATIVE_INFINITY){
                    sum += Math.log10(1 - getFeatureProb(feature, label));
                
                }
            }
        }
        return sum;
    }

    /**
     * Decides to run the only positive or all features probability function
     * This is based on the onlyPos boolean
     * @param ex
     * @param label
     * @return
     */
    private double getLogProb(Example ex, double label){
        // if we are only using positive features
        if(onlyPos){
            return getLogProbPos(ex, label);
        }
        // if we are using all features
        else{
            return getLogProbAll(ex, label);
        }
    }

    /**
     * Calculates the probability of a given feature and label.
     * Gets the count of the feature + lambda smoothing and divides by the amount of times
     * the label is seen, which is also lamba smoothed.
     * Returns the fraction (probability)
     * @param featureIndex
     * @param label
     * @return
     */
    private double getFeatureProb(int featureIndex, double label){
        double numerator = counts.get(label).get(featureIndex) + lambda;
        double denominator = labelCounts.get(label) + lambda * 2;

        return (double) (numerator/denominator);
    }
        

    /**
     * Classifies the example by comparing probabilities of examples with all of the labels
     * and chosing the label with the greatest probability.
     * @param example
     */
    @Override
    public double classify(Example example) {

        double max_probability = Double.NEGATIVE_INFINITY;
        double max_label = Double.NEGATIVE_INFINITY;

        for(double label : labelCounts.keySet()){

            double probability = getLogProb(example, label);
        
            if(probability > max_probability){
                max_probability = probability;
                max_label = label;
            }
        }

        return max_label;
    }

    /**
     * Calculates the confidence by gtting the log probabiltiy of a given example
     * and the predicted label
     * @param example
     */
    @Override
    public double confidence(Example example) {
        return getLogProb(example, classify(example));
    }
    
}