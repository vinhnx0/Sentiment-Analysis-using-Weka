package org.example;

import weka.classifiers.trees.RandomForest;
import weka.classifiers.Evaluation;
import weka.core.Instances;

import java.util.Random;

public class RandomForestEva {
    public String run(Instances data) throws Exception {
        RandomForest model = new RandomForest();
        Evaluation eval = new Evaluation(data);
        eval.crossValidateModel(model, data, 10, new Random(1));

        StringBuilder results = new StringBuilder();
        results.append("===== Random Forest Evaluation =====\n");
        results.append(eval.toSummaryString("\n=== Summary ===\n", false));
        results.append(eval.toClassDetailsString("\n=== Detailed Accuracy By Class ===\n"));
        results.append(eval.toMatrixString("\n=== Confusion Matrix ===\n"));
        return results.toString();
    }
}
