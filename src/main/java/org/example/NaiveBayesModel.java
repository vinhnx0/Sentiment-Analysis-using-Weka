package org.example;

import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.Evaluation;
import weka.core.Instances;

public class NaiveBayesModel {
    public String run(Instances data) throws Exception {
        NaiveBayes nb = new NaiveBayes();
        nb.buildClassifier(data);

        Evaluation eval = new Evaluation(data);
        eval.evaluateModel(nb, data);

        StringBuilder results = new StringBuilder();
        results.append("Results of Naive Bayes Model:\n");
        results.append(nb.getCapabilities().toString());
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
