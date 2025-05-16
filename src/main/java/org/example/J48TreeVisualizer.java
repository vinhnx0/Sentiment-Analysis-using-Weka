package org.example;

import weka.classifiers.trees.J48;
import weka.core.Instances;
import weka.gui.treevisualizer.PlaceNode2;
import weka.gui.treevisualizer.TreeVisualizer;

import javax.swing.*;
import java.awt.*;

public class J48TreeVisualizer {

    public void run(Instances data) throws Exception {
        data.setClassIndex(data.numAttributes() - 1);

        J48 tree = new J48();
        tree.buildClassifier(data);

        final JFrame frame = new JFrame("J48 Decision Tree");
        frame.setSize(800, 600);
        frame.setDefaultCloseOperation(JFrame.DISPOSE_ON_CLOSE);

        TreeVisualizer tv = new TreeVisualizer(null, tree.graph(), new PlaceNode2());
        frame.setLayout(new BorderLayout());
        frame.add(tv, BorderLayout.CENTER);
        frame.setVisible(true);

        tv.fitToScreen();
    }
}
