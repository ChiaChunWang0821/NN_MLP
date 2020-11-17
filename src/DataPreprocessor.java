import java.io.File;
import java.io.FileNotFoundException;
import java.util.Collections;
import java.util.Scanner;

public class DataPreprocessor {

    protected static DataSet readFile(File file){
        DataSet dataSet = new DataSet();

        try {
            Scanner scanner = new Scanner(file);
            while (scanner.hasNextLine()){
                String[] strings = {"0", scanner.nextLine()};
                dataSet.data.add(strings);
                dataSet.totalSet++;
            }

            String string = dataSet.data.get(0)[1];
            String[] s = string.split("\\s+");
            dataSet.dimension = s.length - 1;
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        }

        return dataSet;
    }

    protected static DataSet splitData(DataSet dataSet){
        dataSet.trainingSet = ((int) Math.round(dataSet.totalSet * dataSet.dataPartitionRate));
        dataSet.testingSet = dataSet.totalSet - dataSet.trainingSet;

        dataSet = setTrainingData(dataSet);
        dataSet = setTestingData(dataSet);

        return dataSet;
    }

    private static DataSet setTrainingData(DataSet dataSet){
        dataSet.trainingX.clear();
        dataSet.trainingD.clear();

        for(int i = 0; i < dataSet.trainingSet; i++){
            int random = (int) (Math.random() * dataSet.totalSet);
            while (!dataSet.data.get(random)[0].equals("0")){
                random = (int) (Math.random() * dataSet.totalSet);
            }
            dataSet.data.get(random)[0] = "1";

            String[] numbers = dataSet.data.get(random)[1].split("\\s+");

            double[] x_tmp = new double[dataSet.dimension + 1];
            x_tmp[0] = -1;
            for(int j = 1; j < x_tmp.length; j++){
                x_tmp[j] = Double.parseDouble(numbers[j - 1]);
            }
            dataSet.trainingX.add(x_tmp);
            dataSet.trainingD.add(Double.parseDouble(numbers[dataSet.dimension]));


            if(!dataSet.d.contains(dataSet.trainingD.get(i))){
                dataSet.d.add(dataSet.trainingD.get(i));
            }
        }

        Collections.sort(dataSet.d);

        return dataSet;
    }

    private static DataSet setTestingData(DataSet dataSet){
        dataSet.testingX.clear();
        dataSet.testingD.clear();

        for(int i = 0; i < dataSet.testingSet; i++){
            String string = null;
            for(int k = 0; k < dataSet.data.size(); k++){
                if(dataSet.data.get(k)[0].equals("0")){
                    string = dataSet.data.get(k)[1];
                    dataSet.data.get(k)[0] = "1";
                    break;
                }
            }
            String[] numbers = string.split("\\s+");

            double[] x_tmp = new double[dataSet.dimension + 1];
            x_tmp[0] = -1;
            for(int j = 1; j < x_tmp.length; j++){
                x_tmp[j] = Double.parseDouble(numbers[j - 1]);
            }
            dataSet.testingX.add(x_tmp);

            dataSet.testingD.add(Double.parseDouble(numbers[dataSet.dimension]));
        }

        return dataSet;
    }
}
