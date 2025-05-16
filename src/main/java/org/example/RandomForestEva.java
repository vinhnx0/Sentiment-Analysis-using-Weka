package org.example;

import weka.classifiers.trees.RandomForest;
import weka.classifiers.Evaluation;
import weka.core.Instances;

import java.util.Random;

public class RandomForestEva {
    private int numTrees = 100;

    public RandomForestEva() {}

    public RandomForestEva(int numTrees) {
        this.numTrees = numTrees;
    }

    public String run(Instances data) throws Exception {
        RandomForest model = new RandomForest();
        model.setOptions(new String[]{"-I", Integer.toString(numTrees)}); // set number of trees

        Evaluation eval = new Evaluation(data);
        eval.crossValidateModel(model, data, 10, new Random(1));

        StringBuilder results = new StringBuilder();
        results.append("===== Random Forest Evaluation (").append(numTrees).append(" trees) =====\n");
        results.append(eval.toSummaryString("\n=== Summary ===\n", false));
        results.append(eval.toClassDetailsString("\n=== Detailed Accuracy By Class ===\n"));
        results.append(eval.toMatrixString("\n=== Confusion Matrix ===\n"));
        return results.toString();
    }
}
