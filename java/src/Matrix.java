public class Matrix {
	static double[][] matrix;

	public Matrix(int rows, int cols) {
		matrix = new double[rows][cols];
	}

	public Matrix(double[][] matrix) {
		this.matrix = matrix;
	}

	public static double[][] toArray(Matrix m) {
		return matrix;	
	}

	public static int rows() {
		return matrix[0].length;
	}

	public static int cols() {
		return matrix.length;
	}

	public static void setValueAt(int row, int col, int value) {
		matrix[row][col] = value;
	}

	public static double valueAt(int row, int col) {
		return matrix[row][col];
	}

	public static Matrix dot(Matrix a, Matrix b) throws Exception {
		if (a.cols() != b.rows()) {
			throw new Exception("Matrix.dot(Matrix a, Matrix b) Exception:\nDimensions do not match. (" + a.cols() + " is not " + b.rows() + ")");
		}
	
		int rows = a.rows();
		int cols = b.cols();
		double length = a.cols();
		double[][] matrix = new double[rows][cols];	

		for (int row = 0; row < rows; row++) {
			for (int col = 0; col < cols; col++) {
				for (int len = 0; len < length; len++) {
					matrix[row][col] += a.valueAt(row, len) * b.valueAt(len, col);
				}
			}
		}

		return new Matrix(matrix);
	}

	public Matrix dot(Matrix b) throws Exception {
		if (this.cols() != b.rows()) {
			throw new Exception("Matrix.dot(Matrix b) Exception:\nDimensions do not match. (" + this.cols() + " is not " + this.rows() + ")");
		}

		int rows = this.rows();
		int cols = b.cols();
		int length = this.cols();
		double[][] matrix = new double[rows][cols];

		for (int row = 0; row < rows; row++) {
			for (int col = 0; col < cols; col++) {
				for (int len = 0; len < length; len++) {
					matrix[row][col] += this.valueAt(row, len) * b.valueAt(len, col);
				}
			}
		} 		

		return new Matrix(matrix);
	}
}
