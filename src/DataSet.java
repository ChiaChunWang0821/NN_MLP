import java.util.ArrayList;

public class DataSet {

    static protected double dataPartitionRate; // for training set

    static protected int dimension;
    static protected int totalSet;
    static protected ArrayList<String[]> data = new ArrayList<>(); // 0 or 1, data -> 0: never use

    static protected int trainingSet;
    static protected int testingSet;
    static protected ArrayList<double[]> trainingX = new ArrayList<>();
    static protected ArrayList<double[]> testingX = new ArrayList<>();
    static protected ArrayList<Double> trainingD = new ArrayList<>();
    static protected ArrayList<Double> testingD = new ArrayList<>();

    static protected ArrayList<Double> d = new ArrayList<>();

    protected DataSet(){
        totalSet = 0;
        dataPartitionRate = 0.67;
    }

}
