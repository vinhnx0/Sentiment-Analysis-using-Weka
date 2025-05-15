// package org.example;

// import weka.core.Instances;
// import weka.core.converters.ConverterUtils.DataSource;
// import weka.filters.Filter;
// import weka.filters.unsupervised.attribute.NumericToNominal;

// import java.io.File;

// public class Main {
//     public static void main(String[] args) throws Exception {
//         try {
//             // Specify the dataset path
//             String datasetPath = "src\\data\\train_preprocessed.arff";

//             // Check if the file exists
//             File file = new File(datasetPath);
//             if (!file.exists()) {
//                 System.err.println("Error: File does not exist at path: " + datasetPath);
//                 return;
//             }

//             // Load dataset
//             DataSource source = new DataSource(datasetPath);
//             Instances data = source.getDataSet();

//             // Check if data was loaded successfully
//             if (data == null) {
//                 System.err.println("Error: Failed to load dataset.");
//                 return;
//             }

//             // Set class index (last attribute)
//             if (data.classIndex() == -1) {
//                 data.setClassIndex(data.numAttributes() - 1);
//             }

//             // Convert numeric class to nominal
//             NumericToNominal filter = new NumericToNominal();
//             filter.setAttributeIndices("last");
//             filter.setInputFormat(data);
//             Instances filteredData = Filter.useFilter(data, filter);

//             // Run and print results for each model
//             System.out.println("===================================");
//             System.out.println("Running Naive Bayes...");
//             Naivebayess nb = new Naivebayess();
//             System.out.println(nb.run(filteredData));

//             System.out.println("===================================");
//             System.out.println("Running Random Forest...");
//             RandomForestModel rf = new RandomForestModel();
//             System.out.println(rf.run(filteredData));

//             System.out.println("===================================");
//             System.out.println("Running Logistic Regression...");
//             LogisticRegressionModel lr = new LogisticRegressionModel();
//             System.out.println(lr.run(filteredData));

//             System.out.println("===================================");
//             System.out.println("Running J48 Decision Tree...");
//             J48DecisionTreeModel tree = new J48DecisionTreeModel();
//             System.out.println(tree.run(filteredData));

//         } catch (Exception e) {
//             System.err.println("An error occurred: " + e.getMessage());
//             e.printStackTrace();
//         }
//     }
// }
