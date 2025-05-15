package org.example;

import weka.classifiers.trees.J48;
import weka.core.Instances;
import weka.gui.treevisualizer.PlaceNode2;
import weka.gui.treevisualizer.TreeVisualizer;

import javax.swing.*;
import java.awt.*;

public class J48TreeVisualizer {

    public void run(Instances data) throws Exception {
        // Huấn luyện mô hình J48
        J48 tree = new J48();
        tree.buildClassifier(data);

        // Tạo cửa sổ hiển thị cây
        final JFrame jf = new JFrame("Weka Classifier Tree Visualizer: J48");
        jf.setDefaultCloseOperation(JFrame.DISPOSE_ON_CLOSE); // Chỉ đóng cửa sổ, không thoát chương trình
        jf.setSize(1000, 700);
        jf.getContentPane().setLayout(new BorderLayout());

        // Tạo TreeVisualizer với cây được sinh từ J48
        TreeVisualizer tv = new TreeVisualizer(null, tree.graph(), new PlaceNode2());
        jf.getContentPane().add(tv, BorderLayout.CENTER);

        // Xử lý khi người dùng đóng cửa sổ
        jf.addWindowListener(new java.awt.event.WindowAdapter() {
            public void windowClosing(java.awt.event.WindowEvent e) {
                jf.dispose();
            }
        });

        // Hiển thị và scale phù hợp
        jf.setVisible(true);
        tv.fitToScreen();
    }
}
