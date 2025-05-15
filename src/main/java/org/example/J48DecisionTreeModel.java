package org.example;

import weka.classifiers.trees.J48;
import weka.classifiers.Evaluation;
import weka.core.Instances;

public class J48DecisionTreeModel {
    public String run(Instances data) throws Exception {
        J48 tree = new J48();
        tree.setUnpruned(false); // Optional: prune the tree
        tree.buildClassifier(data);

        Evaluation eval = new Evaluation(data);
        eval.evaluateModel(tree, data);

        StringBuilder results = new StringBuilder();
        results.append("Results of J48 Decision Tree Model:\n");
        results.append(tree.getCapabilities().toString());
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
