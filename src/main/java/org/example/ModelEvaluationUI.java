package org.example;

import javax.swing.*;
import java.awt.*;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.NumericToNominal;
import weka.filters.unsupervised.attribute.PrincipalComponents;

public class ModelEvaluationUI extends JFrame {
    private JTextArea textArea;
    private Instances dataset;

    private final String dataPath = "src\\data\\train_preprocessed.arff";

    public ModelEvaluationUI() {
        setTitle("ML Model Runner and Evaluator");
        setSize(1300, 800);
        setDefaultCloseOperation(EXIT_ON_CLOSE);
        setLocationRelativeTo(null);

        JPanel panel = new JPanel(new BorderLayout());
        textArea = new JTextArea();
        textArea.setEditable(false);
        textArea.setFont(new Font("Monospaced", Font.BOLD, 14));
        JScrollPane scrollPane = new JScrollPane(textArea);
        panel.add(scrollPane, BorderLayout.CENTER);

        JPanel buttonPanel = new JPanel(new GridLayout(6, 2, 10, 10));

        JButton loadDataButton = new JButton("Load Data");
        loadDataButton.addActionListener(e -> {
            try {
                long start = System.currentTimeMillis();

                DataSource source = new DataSource(dataPath);
                Instances data = source.getDataSet();
                if (data.classIndex() == -1) {
                    data.setClassIndex(data.numAttributes() - 1);
                }

                NumericToNominal nominal = new NumericToNominal();
                nominal.setAttributeIndices("last");
                nominal.setInputFormat(data);
                data = Filter.useFilter(data, nominal);

                PrincipalComponents pca = new PrincipalComponents();
                pca.setMaximumAttributes(50);
                pca.setInputFormat(data);
                dataset = Filter.useFilter(data, pca);

                long end = System.currentTimeMillis();
                textArea.append("\nData loaded and reduced to 50 features successfully (" + (end - start) + "ms)\n");
            } catch (Exception ex) {
                ex.printStackTrace();
                textArea.append("\nError loading data!\n");
            }
        });

        JButton runNB = new JButton("Run Naive Bayes");
        runNB.addActionListener(e -> runModelInBackground("NaiveBayes", runNB));

        JButton evalNB = new JButton("Evaluate Naive Bayes");
        evalNB.addActionListener(e -> evalModelInBackground("NaiveBayes", evalNB));

        JButton runRF = new JButton("Run Random Forest");
        runRF.addActionListener(e -> runModelInBackground("RandomForest", runRF));

        JButton evalRF = new JButton("Evaluate Random Forest");
        evalRF.addActionListener(e -> evalModelInBackground("RandomForest", evalRF));

        JButton runLR = new JButton("Run Logistic Regression");
        runLR.addActionListener(e -> runModelInBackground("LogisticRegression", runLR));

        JButton evalLR = new JButton("Evaluate Logistic Regression");
        evalLR.addActionListener(e -> evalModelInBackground("LogisticRegression", evalLR));

        JButton runJ48 = new JButton("Run J48 Decision Tree");
        runJ48.addActionListener(e -> runModelInBackground("J48", runJ48));

        JButton evalJ48 = new JButton("Evaluate J48 Decision Tree");
        evalJ48.addActionListener(e -> evalModelInBackground("J48", evalJ48));

        JButton vizJ48 = new JButton("Visualize J48 Tree");
        vizJ48.addActionListener(e -> {
            if (dataset != null) {
                vizJ48.setEnabled(false);
                textArea.append("\nVisualizing J48 Tree...\n");

                SwingWorker<Void, Void> vizWorker = new SwingWorker<Void, Void>() {
                    @Override
                    protected Void doInBackground() throws Exception {
                        new J48TreeVisualizer().run(dataset);
                        return null;
                    }

                    @Override
                    protected void done() {
                        vizJ48.setEnabled(true);
                    }
                };
                vizWorker.execute();
            } else {
                textArea.append("\nPlease load data first!\n");
            }
        });

        buttonPanel.add(loadDataButton);
        buttonPanel.add(runNB);
        buttonPanel.add(evalNB);
        buttonPanel.add(runRF);
        buttonPanel.add(evalRF);
        buttonPanel.add(runLR);
        buttonPanel.add(evalLR);
        buttonPanel.add(runJ48);
        buttonPanel.add(evalJ48);
        buttonPanel.add(vizJ48);

        panel.add(buttonPanel, BorderLayout.SOUTH);
        add(panel);
    }

    private void runModelInBackground(final String modelName, final JButton button) {
        if (dataset == null) {
            textArea.append("\nPlease load data first!\n");
            return;
        }
        button.setEnabled(false);
        textArea.append("\nRunning " + modelName + "...\n");

        SwingWorker<String, Void> worker = new SwingWorker<String, Void>() {
            @Override
            protected String doInBackground() throws Exception {
                long start = System.currentTimeMillis();
                String result = "";

                if ("NaiveBayes".equals(modelName)) {
                    result = new NaiveBayesModel().run(dataset);
                } else if ("RandomForest".equals(modelName)) {
                    result = new RandomForestModel(50).run(dataset); // 50 trees
                } else if ("LogisticRegression".equals(modelName)) {
                    result = new LogisticRegressionModel(100).run(dataset); // 100 iterations
                } else if ("J48".equals(modelName)) {
                    result = new J48DecisionTreeModel().run(dataset);
                } else {
                    result = "Unknown model\n";
                }

                long end = System.currentTimeMillis();
                return result + "\nTime taken: " + (end - start) + "ms\n";
            }

            @Override
            protected void done() {
                try {
                    textArea.append(get());
                } catch (Exception e) {
                    e.printStackTrace();
                    textArea.append("\nError running model!\n");
                } finally {
                    button.setEnabled(true);
                }
            }
        };
        worker.execute();
    }

    private void evalModelInBackground(final String modelName, final JButton button) {
        if (dataset == null) {
            textArea.append("\nPlease load data first!\n");
            return;
        }
        button.setEnabled(false);
        textArea.append("\nEvaluating " + modelName + "...\n");

        SwingWorker<String, Void> worker = new SwingWorker<String, Void>() {
            @Override
            protected String doInBackground() throws Exception {
                long start = System.currentTimeMillis();
                String result = "";

                if ("NaiveBayes".equals(modelName)) {
                    result = new NaiveBayesEva().run(dataset);
                } else if ("RandomForest".equals(modelName)) {
                    result = new RandomForestEva(50).run(dataset); // 50 trees
                } else if ("LogisticRegression".equals(modelName)) {
                    result = new LogisticRegressionEva(100).run(dataset); // 100 iterations
                } else if ("J48".equals(modelName)) {
                    result = new J48DecisionTreeEva().run(dataset);
                } else {
                    result = "Unknown evaluator\n";
                }

                long end = System.currentTimeMillis();
                return result + "\nEvaluation time: " + (end - start) + "ms\n";
            }

            @Override
            protected void done() {
                try {
                    textArea.append(get());
                } catch (Exception e) {
                    e.printStackTrace();
                    textArea.append("\nError evaluating model!\n");
                } finally {
                    button.setEnabled(true);
                }
            }
        };
        worker.execute();
    }

    public static void main(String[] args) {
        SwingUtilities.invokeLater(new Runnable() {
            public void run() {
                new ModelEvaluationUI().setVisible(true);
            }
        });
    }
}
