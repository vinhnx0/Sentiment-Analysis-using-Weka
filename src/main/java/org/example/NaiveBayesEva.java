package org.example;

import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.Evaluation;
import weka.core.Instances;

import java.util.Random;

public class NaiveBayesEva {
    public String run(Instances data) throws Exception {
        NaiveBayes model = new NaiveBayes();
        Evaluation eval = new Evaluation(data);
        eval.crossValidateModel(model, data, 10, new Random(1));

        StringBuilder results = new StringBuilder();
        results.append("===== Naive Bayes Evaluation =====\n");
        results.append(eval.toSummaryString("\n=== Summary ===\n", false));
        results.append(eval.toClassDetailsString("\n=== Detailed Accuracy By Class ===\n"));
        results.append(eval.toMatrixString("\n=== Confusion Matrix ===\n"));
        return results.toString();
    }
}
