import java.util.HashMap;

public class MLP {

	private int activationFunction = 0;
	private int learningType;
	private Matrix x;
	private Matrix y;
	private HashMap labels = new HashMap();
	private int layers = 0;
	private int[] shape;
	private double lr = 0.001;
	private double desiredAccuracy = .05;
	private double momentum = .01;
	private Matrix prevDelta;
	private Matrix currentError;
	private Matrix currentOutput;
	private int totalEpochs = 0;
	private int trainingData = 0;
	private int batchSize = 100;
	private Matrix weights;
	private Matrix neurons;
	private Matrix biases;

	public MLP(String[] args) {
	
	}

	private void initTrainingData(String trainingFile, int size) {
		
	}

	private void yOneHot(int numLabels, int[] y) {

	}

	private void parseArgs(String[] args) {

	}

	private void initWeights() {

	}

	private void initNeurons() {

	}

	private void initBiases() {

	}

	private void initPersistent() {

	}

	private void forwardPass() {

	}

	private void backwardPass() {

	}

	public static void train() {

	}

	public static void main(String[] args) {
		MLP net = new MLP(args);
		net.train();
	}
}
