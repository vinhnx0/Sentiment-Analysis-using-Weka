package org.example;

import weka.classifiers.trees.J48;
import weka.classifiers.Evaluation;
import weka.core.Instances;

import java.util.Random;

public class J48DecisionTreeEva {
    public String run(Instances data) throws Exception {
        J48 model = new J48();
        Evaluation eval = new Evaluation(data);
        eval.crossValidateModel(model, data, 10, new Random(1));

        StringBuilder results = new StringBuilder();
        results.append("===== J48 Decision Tree Evaluation =====\n");
        results.append(eval.toSummaryString("\n=== Summary ===\n", false));
        results.append(eval.toClassDetailsString("\n=== Detailed Accuracy By Class ===\n"));
        results.append(eval.toMatrixString("\n=== Confusion Matrix ===\n"));
        return results.toString();
    }
}
