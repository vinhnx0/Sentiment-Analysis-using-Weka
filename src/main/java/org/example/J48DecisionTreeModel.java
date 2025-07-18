package org.example;

import weka.classifiers.trees.J48;
import weka.classifiers.Evaluation;
import weka.core.Instances;

public class J48DecisionTreeModel {
    public String run(Instances data) throws Exception {
        J48 tree = new J48();
        tree.setOptions(new String[]{"-C", "0.25", "-M", "2"}); // Confidence factor & min instances per leaf
        tree.buildClassifier(data);

        Evaluation eval = new Evaluation(data);
        eval.evaluateModel(tree, data);

        StringBuilder results = new StringBuilder();
        results.append("Results of J48 Decision Tree:\n");
        results.append(tree.getCapabilities().toString());
        results.append(String.format("Correct: %.2f%%\n", eval.pctCorrect()));
        results.append(String.format("Incorrect: %.2f%%\n", eval.pctIncorrect()));
        results.append(String.format("Precision: %.2f%%\n", eval.precision(1) * 100));
        results.append(String.format("Recall: %.2f%%\n", eval.recall(1) * 100));
        results.append(String.format("F1 Score: %.2f%%\n", eval.fMeasure(1) * 100));
        results.append(String.format("AUC: %.2f\n", eval.areaUnderROC(1)));
        results.append(eval.toMatrixString("=== Overall Confusion Matrix ===\n"));
        return results.toString();
    }
}
