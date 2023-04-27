package main.java.org.example.lesson;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;

import it.uniroma2.sag.kelp.data.dataset.SimpleDataset;
import it.uniroma2.sag.kelp.data.example.Example;
import it.uniroma2.sag.kelp.data.example.ExampleFactory;
import it.uniroma2.sag.kelp.data.example.ParsingExampleException;
import it.uniroma2.sag.kelp.data.label.Label;
import it.uniroma2.sag.kelp.data.representation.Representation;
import it.uniroma2.sag.kelp.kernel.Kernel;
import it.uniroma2.sag.kelp.kernel.cache.FixSizeKernelCache;
import it.uniroma2.sag.kelp.kernel.cache.KernelCache;
import it.uniroma2.sag.kelp.kernel.standard.LinearKernelCombination;
import it.uniroma2.sag.kelp.kernel.standard.NormalizationKernel;
import it.uniroma2.sag.kelp.kernel.standard.PolynomialKernel;
import it.uniroma2.sag.kelp.kernel.tree.SubSetTreeKernel;
import it.uniroma2.sag.kelp.kernel.vector.LinearKernel;
import it.uniroma2.sag.kelp.learningalgorithm.classification.libsvm.BinaryCSvmClassification;
import it.uniroma2.sag.kelp.learningalgorithm.classification.multiclassification.OneVsAllLearning;
import it.uniroma2.sag.kelp.predictionfunction.classifier.ClassificationOutput;
import it.uniroma2.sag.kelp.predictionfunction.classifier.Classifier;
import it.uniroma2.sag.kelp.utils.JacksonSerializerWrapper;
import it.uniroma2.sag.kelp.utils.evaluation.MulticlassClassificationEvaluator;

public class Main {

	public static void main(String[] args) throws Exception {
		
		// Suppress all logs but warnings
		System.setProperty("org.slf4j.simpleLogger.defaultLogLevel", "error");
		
		boolean verbose = false;
		
		String trainingSetFilePath = "src/main/resources/qc_train.klp";
		String testsetFilePath = "src/main/resources/qc_test.klp";
		
		// SVM params
		float c = 1f;
		
		// kernelType = {linear, linear_sbert, polynomial, tree_kernel, linear_tree_kernel, linear_tree_normalized_kernel}
		String kernelType = "linear_sbert";
		
		// Degree if the kernel chosen is polynomial
		int degree = 5;
		
		// Read the training and test dataset
		SimpleDataset trainingSet = new SimpleDataset();
		trainingSet.populate(trainingSetFilePath);
		System.out.println("The training set is made of " + trainingSet.getNumberOfExamples() + " examples.");

		SimpleDataset testSet = new SimpleDataset();
		testSet.populate(testsetFilePath);
		System.out.println("The test set is made of " + testSet.getNumberOfExamples() + " examples.");

		System.out.println("********************************************");
		// Print the number of train and test examples for each class
		for (Label l : trainingSet.getClassificationLabels()) {
			System.out.println("Positive training examples for the class " + l.toString() + " "
					+ trainingSet.getNumberOfPositiveExamples(l));
			System.out.println("Negative training examples for the class " + l.toString() + " "
					+ trainingSet.getNumberOfNegativeExamples(l));
			System.out.println("********************************************");
		}
		
		// Calculating the size of the gram matrix to store all the examples
		int cacheSize = trainingSet.getNumberOfExamples() + testSet.getNumberOfExamples();
		
		// Initialize the proper kernel function
		Kernel usedKernel = null;
		if (kernelType.equalsIgnoreCase("linear")) {
			String vectorRepresentationName = "bow";
			Kernel linearKernel = new LinearKernel(vectorRepresentationName);
			usedKernel = linearKernel;
		} else if (kernelType.equalsIgnoreCase("linear_sbert")) {
			String vectorRepresentationName = "sbert";
			Kernel linearKernel = new LinearKernel(vectorRepresentationName);
			usedKernel = linearKernel;
			// add here the sbert vector representation from file to the datasets
			addSBertRepresentation(vectorRepresentationName, trainingSet, testSet);
		} else if (kernelType.equalsIgnoreCase("polynomial")) {
			String vectorRepresentationName = "bow";
			Kernel linearKernel = new LinearKernel(vectorRepresentationName);
			Kernel polynomialKernel = new PolynomialKernel(degree, linearKernel);
			usedKernel = polynomialKernel;
		} else if (kernelType.equalsIgnoreCase("tree_kernel")) {
			String treeRepresentationName = "grct";
			float lambda = 0.4f;
			Kernel tkgrct = new SubSetTreeKernel(lambda, treeRepresentationName);
			usedKernel = tkgrct;
		} else if (kernelType.equalsIgnoreCase("linear_tree_kernel")) {
			String vectorRepresentationName = "bow";
			String treeRepresentationName = "grct";
			float lambda = 0.4f;

			Kernel linearKernel = new LinearKernel(vectorRepresentationName);
			Kernel tkgrct = new SubSetTreeKernel(lambda, treeRepresentationName);

			LinearKernelCombination combination = new LinearKernelCombination();
			combination.addKernel(1, linearKernel);
			combination.addKernel(1, tkgrct);
			usedKernel = combination;
		} else if (kernelType.equalsIgnoreCase("linear_tree_normalized_kernel")) {
			String vectorRepresentationName = "bow";
			String treeRepresentationName = "grct";
			float lambda = 0.4f;

			Kernel linearKernel = new LinearKernel(vectorRepresentationName);
			Kernel normalizedLinearKernel = new NormalizationKernel(linearKernel);
			Kernel treeKernel = new SubSetTreeKernel(lambda, treeRepresentationName);
			Kernel normalizedTreeKernel = new NormalizationKernel(treeKernel);

			LinearKernelCombination combination = new LinearKernelCombination();
			combination.addKernel(1, normalizedLinearKernel);
			combination.addKernel(1, normalizedTreeKernel);
			usedKernel = combination;
		} else {
			System.err.println("The specified kernel (" + kernelType + ") is not valid.");
		}
		
		// Setting the cache to speed up the computations
		KernelCache cache=new FixSizeKernelCache(cacheSize);
		usedKernel.setKernelCache(cache);
		
		// Instantiate the SVM learning Algorithm. 
		BinaryCSvmClassification svmSolver = new BinaryCSvmClassification();
		//Set the kernel
		svmSolver.setKernel(usedKernel);
		//Set the C parameter
		svmSolver.setCn(c);
		svmSolver.setCp(c);
		
		// Instantiate the multi-class classifier that apply a One-vs-All schema
		OneVsAllLearning ovaLearner = new OneVsAllLearning();
		ovaLearner.setBaseAlgorithm(svmSolver);
		ovaLearner.setLabels(trainingSet.getClassificationLabels());
		// Writing the learning algorithm and the kernel to file
		JacksonSerializerWrapper serializer = new JacksonSerializerWrapper();
		serializer.writeValueOnFile(ovaLearner, "ova_learning_algorithm.klp");

		// Learn and get the prediction function
		ovaLearner.learn(trainingSet);
		// Selecting the prediction function 
		Classifier classifier = ovaLearner.getPredictionFunction();
		// Write the model (aka the Classifier for further use)
		if(kernelType.equalsIgnoreCase("polynomial")) {
			kernelType = kernelType + "_degree" + degree;
		}
		serializer.writeValueOnFile(classifier, "model_kernel-" + kernelType + "_cp" + c + "_cn" + c + ".klp");

		// Building the evaluation function
		MulticlassClassificationEvaluator evaluator = new MulticlassClassificationEvaluator(
				trainingSet.getClassificationLabels());

		// Classify examples and compute the accuracy
		for (Example e : testSet.getExamples()) {
			// Predict the class
			ClassificationOutput p = classifier.predict(e);
			evaluator.addCount(e, p);
			if(verbose) {
				System.out.println("Question:\t" + e.getRepresentation("quest"));
				System.out.println("Original class:\t" + e.getClassificationLabels());
				System.out.println("Predicted class:\t" + p.getPredictedClasses());
				System.out.println();
			}
		}

		System.out.println("Kernel: "+ kernelType + "\tAccuracy: " + evaluator.getAccuracy());
	}
	
	public static void addSBertRepresentation(String vectorRepresentationName, SimpleDataset trainingSet, SimpleDataset testSet) throws InstantiationException, ParsingExampleException {
		// read training and test txt file
		ArrayList<String> sbertTraining = readTxtFile("src/main/resources/train_embeddings_sbert.txt");
		
		// loop through training set, get the sbert vector by index and add the representation to the example
		for (int i = 0; i < trainingSet.getExamples().size(); i++) {
			Example example = trainingSet.getExamples().get(i);
			String sbertVector = sbertTraining.get(i);
			Representation representation = ExampleFactory.parseExample("|BDV:sbert| " + sbertVector + " |EDV|").getRepresentation("sbert");
			example.addRepresentation(vectorRepresentationName, representation);
		}

		ArrayList<String> sbertTest = readTxtFile("src/main/resources/test_embeddings_sbert.txt");
		for (int i = 0; i < testSet.getExamples().size(); i++) {
			Example example = testSet.getExamples().get(i);
			String sbertVector = sbertTest.get(i);
			Representation representation = ExampleFactory.parseExample("|BDV:sbert| " + sbertVector + " |EDV|").getRepresentation("sbert");
			example.addRepresentation(vectorRepresentationName, representation);
		}
	}
	
	public static ArrayList<String> readTxtFile(String filename) {
		ArrayList<String> list = new ArrayList<>();
		BufferedReader reader;

		try {
			reader = new BufferedReader(new FileReader(filename));
			String line = reader.readLine();

			while (line != null) {
				list.add(line);
				// read next line
				line = reader.readLine();
			}

			reader.close();
		} catch (IOException e) {
			e.printStackTrace();
		}
		
		return list;
	}

}
