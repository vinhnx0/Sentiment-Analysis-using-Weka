package org.example;
import weka.core.Instances;
import weka.core.converters.ArffSaver;
import weka.core.converters.CSVLoader;
import java.io.File;
public class CSV2aff {
    public static void main(String[] args) throws Exception {
        CSVLoader loader = new CSVLoader();
        loader.setSource(new File("src/data/train_preprocessed.csv")); //path file
        Instances data = loader.getDataSet();

        ArffSaver saver = new ArffSaver();
        saver.setInstances(data);

        saver.setFile(new File("src/data/train.arff"));
        saver.writeBatch();
    }
}
