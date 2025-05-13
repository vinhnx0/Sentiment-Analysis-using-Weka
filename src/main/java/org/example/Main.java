package org.example;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import java.io.File;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.NumericToNominal;//TIP To <b>Run</b> code, press <shortcut actionId="Run"/> or
public class Main {
    public static void main(String[] args) throws Exception{//TIP Press <shortcut actionId="ShowIntentionActions"/> with your caret at the highlighted text
        try {
            // Specify the dataset path
            String datasetPath = "E:\\HCMIU\\Lab\\DM\\project\\data_mining\\src\\data\\train_preprocessed.arff";

            // Check if the file exists
            File file = new File(datasetPath);
            if (!file.exists()) {
                System.err.println("Error: File does not exist at path: " + datasetPath);
                return;
            }

            // Load dataset
            DataSource source = new DataSource(datasetPath);
            Instances data = source.getDataSet();

            // Check if data was loaded successfully
            if (data == null) {
                System.err.println("Error: Failed to load dataset.");
                return;
            }

            // Set class index
            if (data.classIndex() == -1) {
                data.setClassIndex(data.numAttributes() - 1);
            }

            // Convert numeric class to nominal
            NumericToNominal filter = new NumericToNominal();
            filter.setAttributeIndices("last"); // Apply to the class attribute (last)
            filter.setInputFormat(data);
            Instances filteredData = Filter.useFilter(data, filter);

            // Initialize Naivebayess and run
            Naivebayess nb = new Naivebayess();
            String results = nb.run(filteredData);

            // Print results
            System.out.println(results);
        } catch (Exception e) {
            System.err.println("An error occurred: " + e.getMessage());
            e.printStackTrace();
        }
    }
}