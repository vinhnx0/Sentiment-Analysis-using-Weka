package org.example;

import weka.classifiers.functions.Logistic;
import weka.classifiers.Evaluation;
import weka.core.Instances;

public class LogisticRegressionModel {
    public String run(Instances data) throws Exception {
        Logistic lr = new Logistic();
        lr.buildClassifier(data);

        Evaluation eval = new Evaluation(data);
        eval.evaluateModel(lr, data);

        StringBuilder results = new StringBuilder();
        results.append("Results of Logistic Regression Model:\n");
        results.append(lr.getCapabilities().toString());
        results.append(String.format("Correct: %.2f%%\n", eval.pctCorrect()));
        results.append(String.format("Incorrect: %.2f%%\n", eval.pctIncorrect()));
        results.append(String.format("Precision: %.2f%%\n", eval.precision(1) * 100));
        results.append(String.format("Recall: %.2f%%\n", eval.recall(1) * 100));
        results.append(String.format("F1 Score: %.2f%%\n", eval.fMeasure(1) * 100));
        results.append(String.format("Error Rate: %.2f%%\n", eval.errorRate() * 100));
        results.append(String.format("AUC: %.2f\n", eval.areaUnderROC(1)));
        results.append(String.format("Kappa: %.2f\n", eval.kappa()));
        results.append(String.format("Mean Absolute Error: %.4f\n", eval.meanAbsoluteError()));
        results.append(String.format("Root Mean Squared Error: %.4f\n", eval.rootMeanSquaredError()));
        results.append(String.format("Relative Absolute Error: %.2f%%\n", eval.relativeAbsoluteError()));
        results.append(String.format("Root Relative Squared Error: %.2f%%\n", eval.rootRelativeSquaredError()));
        results.append(eval.toMatrixString("=== Overall Confusion Matrix ===\n"));

        return results.toString();
    }
}
