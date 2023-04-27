package main.java.org.example.lesson;

import java.io.File;

import it.uniroma2.sag.kelp.data.dataset.SimpleDataset;
import it.uniroma2.sag.kelp.data.example.Example;
import it.uniroma2.sag.kelp.learningalgorithm.classification.multiclassification.OneVsAllLearning;
import it.uniroma2.sag.kelp.predictionfunction.classifier.ClassificationOutput;
import it.uniroma2.sag.kelp.predictionfunction.classifier.Classifier;
import it.uniroma2.sag.kelp.utils.JacksonSerializerWrapper;
import it.uniroma2.sag.kelp.utils.evaluation.MulticlassClassificationEvaluator;

public class WmIRQuestionClassificationExampleFromJson {
	public static void main(String[] args) throws Exception {

		if (args.length != 3) {
			System.err.println("Usage: training_set_path test_set_path json_file_path");
			return;
		}
		String trainingSetFilePath = args[0];
		String testsetFilePath = args[1];
		String jsonAlgorithmPath = args[2];

		// Read the training and test dataset
		SimpleDataset trainingSet = new SimpleDataset();
		trainingSet.populate(trainingSetFilePath);

		SimpleDataset testSet = new SimpleDataset();
		testSet.populate(testsetFilePath);

		// Loading the classifier from the JSON file
		JacksonSerializerWrapper serializer = new JacksonSerializerWrapper();
		OneVsAllLearning ovaLearner = serializer.readValue(new File(jsonAlgorithmPath), OneVsAllLearning.class);
		ovaLearner.setLabels(trainingSet.getClassificationLabels());

		// Learn and get the prediction function
		ovaLearner.learn(trainingSet);
		Classifier f = ovaLearner.getPredictionFunction();

		// Classify examples and compute the accuracy
		MulticlassClassificationEvaluator evaluator = new MulticlassClassificationEvaluator(
				trainingSet.getClassificationLabels());
		for (Example e : testSet.getExamples()) {
			// Predict the class
			ClassificationOutput p = f.predict(testSet.getNextExample());
			evaluator.addCount(e, p);
			System.out.println("Question:\t" + e.getRepresentation("quest"));
			System.out.println("Original class:\t" + e.getClassificationLabels());
			System.out.println("Predicted class:\t" + p.getPredictedClasses());
			System.out.println();
		}
		System.out.println("Accuracy: " + evaluator.getAccuracy());
	}

}
