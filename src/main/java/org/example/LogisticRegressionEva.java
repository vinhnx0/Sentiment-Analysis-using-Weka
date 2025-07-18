package org.example;

import weka.classifiers.functions.Logistic;
import weka.classifiers.Evaluation;
import weka.core.Instances;

import java.util.Random;

public class LogisticRegressionEva {
    private int maxIters = 100;

    public LogisticRegressionEva() {}

    public LogisticRegressionEva(int maxIters) {
        this.maxIters = maxIters;
    }

    public String run(Instances data) throws Exception {
        Logistic model = new Logistic();
        model.setMaxIts(maxIters);

        Evaluation eval = new Evaluation(data);
        eval.crossValidateModel(model, data, 10, new Random(1));

        StringBuilder results = new StringBuilder();
        results.append("===== Logistic Regression Evaluation (maxIters = ").append(maxIters).append(") =====\n");
        results.append(eval.toSummaryString("\n=== Summary ===\n", false));
        results.append(eval.toClassDetailsString("\n=== Detailed Accuracy By Class ===\n"));
        results.append(eval.toMatrixString("\n=== Confusion Matrix ===\n"));
        return results.toString();
    }
}
    