# Sentiment Analysis using Weka

This project focuses on analyzing emotions in text data using data mining techniques with the Weka library. The goal is to classify emotional content within the Emotions Analysis dataset, which consists of text samples labeled with specific emotions such as sadness, joy, love, anger, fear, and surprise. By leveraging advanced computational methods, the project seeks to uncover patterns in how emotions are conveyed in social media texts.

## Project Overview

The primary objective of this project is to develop a data mining framework that employs classification models to accurately predict the emotion conveyed in text. This involves building and evaluating predictive models to categorize emotions effectively. The framework aims to achieve high accuracy in emotion detection, supporting applications in various domains such as sentiment analysis and social media monitoring.

## Dataset

The dataset used in this project is the Emotions Analysis dataset, sourced from [Kaggle](https://www.kaggle.com/code/abdmental01/emotions-analysis-gru-94/input). It contains Twitter comments labeled with emotions and includes the following attributes:
- `id`: Unique identifier for each tweet (Integer).
- `text`: Tweet content (String).
- `label`: Emotion label (0–5) representing sadness (0), joy (1), love (2), anger (3), fear (4), and surprise (5) (Integer).

The dataset is preprocessed into `train_preprocessed.csv` (10,000 instances) and `test_preprocessed.csv` (2,000 instances), with text transformed into embeddings using the all-MiniLM-L6-v2 model, and further converted to ARFF format for Weka compatibility.

## Project Structure

```
Sentiment-Analysis-using-Weka/
├── .idea/                          # IDE configuration files
├── src/                            # Source code directory
│    ├── data/                      # Data files
│         └── train_preprocessed.arff  # Preprocessed training data in ARFF format
│    ├── main/java/org/example/     # Java source files
│         ├── J48DecisionTreeEva.java
│         ├── J48DecisionTreeModel.java
│         ├── J48TreeVisualizer.java
│         ├── LogisticRegressionEva.java
│         ├── LogisticRegressionModel.java
│         ├── Main.java
│         ├── ModelEvaluationUI.java
│         ├── NaiveBayesEva.java
│         ├── NaiveBayesModel.java
│         ├── RandomForestEva.java
│         └── RandomForestModel.java
├── weka.jar
├── mtj-1.0.4.jar
├── .gitignore                      # Git ignore file
├── README.md                       # Project README file
├── data.csv                        # Raw dataset
├── pom.xml                         # Maven project configuration
├── preprocessing.ipynb             # Jupyter notebook for data preprocessing
├── test_preprocessed.csv           # Preprocessed test data
└── train_preprocessed.csv          # Preprocessed training data
```

## Installation

To run this project, ensure you have the following installed:
- **Java Development Kit (JDK)**: Version 8 or higher
- **Maven**: For dependency management
- **Weka**: Included via Maven dependencies
- **Python**: Required for preprocessing (optional), with libraries like pandas and sentence-transformers

### Steps to Set Up

1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-username/Sentiment-Analysis-using-Weka.git
   cd Sentiment-Analysis-using-Weka
   ```

2. **Install dependencies**:
   ```bash
   mvn clean install
   ```

3. **(Optional) Preprocess the data**:
   - Install Python and required libraries (`pip install pandas sentence-transformers`).
   - Run the `preprocessing.ipynb` Jupyter notebook to generate `train_preprocessed.csv`, `test_preprocessed.csv`, and `train_preprocessed.arff`.

## Adding JAR Dependencies in IntelliJ IDEA

This guide explains how to add the `weka.jar` and `mtj-1.0.4.jar` files as dependencies in IntelliJ IDEA for the **Sentiment-Analysis-using-Weka** project.

### Steps to Add JAR Dependencies in IntelliJ IDEA

1. **Open the Project in IntelliJ IDEA**:

   - Launch IntelliJ IDEA and open the `Sentiment-Analysis-using-Weka` project.

2. **Access Project Structure**:

   - Go to `File` &gt; `Project Structure` (or press `Ctrl+Alt+Shift+S` on Windows/Linux, `Cmd+;` on macOS).

3. **Add JAR Files to Libraries**:

   - In the `Project Structure` window, select `Modules` from the left panel.
   - Select the module (e.g., `Sentiment-Analysis-using-Weka`).
   - Go to the `Dependencies` tab and click the `+` button (or `Add` button) at the bottom.
   - Choose `JARs or Directories`.
   - Navigate to the project root directory and select both `weka.jar` and `mtj-1.0.4.jar`.
   - Click `OK` to add them to the module's dependencies.

4. **Set Scope**:

   - Ensure the scope for both JARs is set to `Compile` (default) to make them available during compilation and runtime.

5. **Apply Changes**:

   - Click `Apply` and then `OK` to close the `Project Structure` window.

6. **Verify Dependencies**:

   - Check the `External Libraries` section in the Project Explorer to confirm that `weka.jar` and `mtj-1.0.4.jar` are listed.
   - Rebuild the project (`Build` &gt; `Rebuild Project`) to ensure no errors occur due to missing dependencies.

### Using Maven (Optional)

If you prefer managing dependencies via Maven, the `pom.xml` file already includes Weka dependencies. Ensure the following dependencies are present:

```xml
<dependency>
    <groupId>nz.ac.waikato.cms.weka</groupId>
    <artifactId>weka-stable</artifactId>
    <version>3.8.6</version>
</dependency>
<dependency>
    <groupId>com.github.fommil.netlib</groupId>
    <artifactId>mtj</artifactId>
    <version>1.0.4</version>
</dependency>
```

Run `mvn clean install` to download and include these dependencies automatically.

### Troubleshooting

- **ClassNotFoundException or NoClassDefFoundError**: Ensure both JARs are correctly added to the module dependencies and the project is rebuilt.
- **Version Conflicts**: Verify that mtj-1.0.4.jar is compatible with weka.jar. If issues arise, consider using the Maven approach to ensure consistent versions.
- **JAR Not Found**: Confirm that weka.jar and mtj-1.0.4.jar are in the project root directory as shown in the structure.

### Additional Notes

- The `weka.jar` file provides the core Weka library for machine learning tasks.
- The `mtj-1.0.4.jar` file is a dependency required by Weka for matrix operations.
- If you modify the project structure (e.g., move JARs to a `lib/` folder), update the paths in IntelliJ IDEA accordingly.

For further assistance, refer to the Weka Documentation or contact the project contributors.

## Usage

### Running the Application

The main entry point is `Main.java`, which launches a GUI for model evaluation. To run it:
```bash
mvn exec:java -Dexec.mainClass="org.example.Main"
```

### Model Evaluation

The project implements four classification models using Weka:
- **J48 Decision Tree**: A decision tree algorithm.
- **Logistic Regression**: A linear model for probabilistic classification.
- **Naive Bayes**: A probabilistic classifier based on Bayes' Theorem.
- **Random Forest**: An ensemble of decision trees.

Each model has evaluation classes (e.g., `J48DecisionTreeEva.java`) that perform 10-fold cross-validation and output performance metrics like accuracy, precision, recall, and F1-score.

### GUI Interface

The `ModelEvaluationUI.java` class provides a graphical interface to:
- Select a model.
- Run evaluations.
- View results (e.g., accuracy, confusion matrix).

## Project Report

The full report, included in the repository, covers:
1. **Introduction**: Project goals and significance.
2. **Data Pre-Processing**: Cleaning and transformation steps.
3. **Classification/Prediction Algorithm**: Model selection and implementation.
4. **Improvement of Results**: Techniques to enhance performance.
5. **Model Evaluation**: Performance metrics and analysis.
6. **Conclusions**: Key findings and future improvements.
7. **References**: Dataset and library sources.

Refer to the report for detailed insights.

## Contributors

- **Nguyễn Xuân Vinh** (ITDSIU21069)
- **Phan Danh Đức** (ITDSIU21012)
- **Lê Nguyễn Thành Long** (ITDSIU21097)
- **Nguyễn Bá Duy** (ITDSIU21014)
- **Phạm Huỳnh Thanh Quân** (ITDSIU21110)

## References

- [Emotions Analysis Dataset](https://www.kaggle.com/code/abdmental01/emotions-analysis-gru-94/input)
- [Weka Documentation](https://weka.sourceforge.io/doc.dev/)
- [RandomForest Class Reference](https://weka.sourceforge.io/doc.dev/weka/classifiers/trees/RandomForest.html)
- [NaiveBayes Class Reference](https://weka.sourceforge.io/doc.dev/weka/classifiers/bayes/NaiveBayes.html)
- [J48 Decision Tree Class Reference](https://weka.sourceforge.io/doc.dev/weka/classifiers/trees/J48.html)
- Hall, M., Frank, E., Holmes, G. & Pfahringer, B., “An Update on the Weka Data Mining Software,” *SIGKDD Explorations*, vol. 11, no. 1, 2009.
- Pedregosa, F. et al., “Scikit-learn: Machine Learning in Python,” *Journal of Machine Learning Research*, vol. 12, pp. 2825–2830, 2011.
- McKinney, W., “pandas: a foundational Python library for data analysis and statistics,” *Python for High Performance and Scientific Computing*, 2011.
- Virtanen, P. et al., “SciPy 1.0: Fundamental Algorithms for Scientific Computing in Python,” *Nature Methods*, vol. 17, pp. 261–272, 2020.
- Oracle, “Concurrency in Swing: SwingWorker,” *Oracle Java™ Tutorials*, 2025.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
