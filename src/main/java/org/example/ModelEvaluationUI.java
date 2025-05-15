package org.example;

import javax.swing.*;
import java.awt.*;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.NumericToNominal;

public class ModelEvaluationUI extends JFrame {
    private JTextArea textArea;
    private Instances dataset;

    private final String dataPath = "src\\data\\train_preprocessed.arff";

    // Models
    private Naivebayess naiveBayess = new Naivebayess();
    private RandomForestModel randomForestModel = new RandomForestModel();
    private LogisticRegressionModel logisticRegressionModel = new LogisticRegressionModel();
    private J48DecisionTreeModel j48DecisionTreeModel = new J48DecisionTreeModel();

    // Evaluation
    private NaiveBayesEva naiveBayesEva = new NaiveBayesEva();
    private RandomForestEva randomForestEva = new RandomForestEva();
    private LogisticRegressionEva logisticRegressionEva = new LogisticRegressionEva();
    private J48DecisionTreeEva j48DecisionTreeEva = new J48DecisionTreeEva();

    // Visualizer
    private J48TreeVisualizer j48TreeVisualizer = new J48TreeVisualizer();

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
                DataSource source = new DataSource(dataPath);
                Instances data = source.getDataSet();
                if (data.classIndex() == -1) {
                    data.setClassIndex(data.numAttributes() - 1);
                }
                NumericToNominal filter = new NumericToNominal();
                filter.setAttributeIndices("last");
                filter.setInputFormat(data);
                dataset = Filter.useFilter(data, filter);
                textArea.append("\nData loaded successfully from: " + dataPath + "\n");
            } catch (Exception ex) {
                ex.printStackTrace();
                textArea.append("\nError loading data!\n");
            }
        });

        JButton runNB = new JButton("Run Naive Bayes");
        runNB.addActionListener(e -> runModelInBackground(naiveBayess, runNB));

        JButton evalNB = new JButton("Evaluate Naive Bayes");
        evalNB.addActionListener(e -> evalModelInBackground(naiveBayesEva, evalNB));

        JButton runRF = new JButton("Run Random Forest");
        runRF.addActionListener(e -> runModelInBackground(randomForestModel, runRF));

        JButton evalRF = new JButton("Evaluate Random Forest");
        evalRF.addActionListener(e -> evalModelInBackground(randomForestEva, evalRF));

        JButton runLR = new JButton("Run Logistic Regression");
        runLR.addActionListener(e -> runModelInBackground(logisticRegressionModel, runLR));

        JButton evalLR = new JButton("Evaluate Logistic Regression");
        evalLR.addActionListener(e -> evalModelInBackground(logisticRegressionEva, evalLR));

        JButton runJ48 = new JButton("Run J48 Decision Tree");
        runJ48.addActionListener(e -> runModelInBackground(j48DecisionTreeModel, runJ48));

        JButton evalJ48 = new JButton("Evaluate J48 Decision Tree");
        evalJ48.addActionListener(e -> evalModelInBackground(j48DecisionTreeEva, evalJ48));

        JButton vizJ48 = new JButton("Visualize J48 Tree");
        vizJ48.addActionListener(e -> {
            if (dataset != null) {
                // Tốt nhất cũng nên chạy background
                vizJ48.setEnabled(false);
                textArea.append("\nVisualizing J48 Tree...\n");
                SwingWorker<Void, Void> vizWorker = new SwingWorker<Void, Void>() {
                    @Override
                    protected Void doInBackground() throws Exception {
                        j48TreeVisualizer.run(dataset);
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

    private void runModelInBackground(Object model, JButton button) {
        if (dataset == null) {
            textArea.append("\nPlease load data first!\n");
            return;
        }
        button.setEnabled(false);
        textArea.append("\nRunning " + model.getClass().getSimpleName() + "...\n");

        SwingWorker<String, Void> worker = new SwingWorker<String, Void>() {
            @Override
            protected String doInBackground() throws Exception {
                if (model instanceof Naivebayess) return ((Naivebayess) model).run(dataset);
                else if (model instanceof RandomForestModel) return ((RandomForestModel) model).run(dataset);
                else if (model instanceof LogisticRegressionModel) return ((LogisticRegressionModel) model).run(dataset);
                else if (model instanceof J48DecisionTreeModel) return ((J48DecisionTreeModel) model).run(dataset);
                else return "Unknown model!\n";
            }

            @Override
            protected void done() {
                try {
                    String result = get();
                    textArea.append(result);
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

    private void evalModelInBackground(Object evaluator, JButton button) {
        if (dataset == null) {
            textArea.append("\nPlease load data first!\n");
            return;
        }
        button.setEnabled(false);
        textArea.append("\nEvaluating " + evaluator.getClass().getSimpleName() + "...\n");

        SwingWorker<String, Void> worker = new SwingWorker<String, Void>() {
            @Override
            protected String doInBackground() throws Exception {
                if (evaluator instanceof NaiveBayesEva) return ((NaiveBayesEva) evaluator).run(dataset);
                else if (evaluator instanceof RandomForestEva) return ((RandomForestEva) evaluator).run(dataset);
                else if (evaluator instanceof LogisticRegressionEva) return ((LogisticRegressionEva) evaluator).run(dataset);
                else if (evaluator instanceof J48DecisionTreeEva) return ((J48DecisionTreeEva) evaluator).run(dataset);
                else return "Unknown evaluator!\n";
            }

            @Override
            protected void done() {
                try {
                    String result = get();
                    textArea.append(result);
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
        SwingUtilities.invokeLater(() -> {
            ModelEvaluationUI app = new ModelEvaluationUI();
            app.setVisible(true);
        });
    }
}
