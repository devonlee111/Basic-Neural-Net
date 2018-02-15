public class Matrix {
	double[][] matrix;

	public Matrix(int rows, int cols) {
		matrix = new double[rows][cols];
	}

	public Matrix(double[][] matrix) {
		this.matrix = matrix;
	}

	public double[][] toArray() {
		return matrix;	
	}

	public int rows() {
		return matrix.length;
	}

	public int cols() {
		return matrix[0].length;
	}

	public void setValueAt(int row, int col, double value) {
		matrix[row][col] = value;
	}

	public double valueAt(int row, int col) {
		return matrix[row][col];
	}

	public Matrix add(Matrix b) {
		if (this.rows() != b.rows() || this.cols() != b.cols()) {
			System.err.println("Matrix.add(Matrix b): Dimensions do not match (" + this.rows() + ", " + this.cols() + " and (" + b.rows() + ", " + b.cols() + ")");
			System.exit(1);
		}

		double[][] matrix = new double[this.rows()][this.cols()];
		for (int row = 0; row < this.rows(); row++) {
			for (int col = 0; col < this.cols(); col++) {
				matrix[row][col] = this.valueAt(row, col) + b.valueAt(row, col);
			}
		}
		
		return new Matrix(matrix);
	}

	public Matrix subtract(Matrix b) {
		if (this.rows() != b.rows() || this.cols() != b.cols()) {
			System.err.println("Matrix.add(Matrix b): Dimensions do not match (" + this.rows() + ", " + this.cols() + " and (" + b.rows() + ", " + b.cols() + ")");
			System.exit(1);
		}

		double[][] matrix = new double[this.rows()][this.cols()];
		for (int row = 0; row < this.rows(); row++) {
			for (int col = 0; col < this.cols(); col++) {
				matrix[row][col] = this.valueAt(row, col) - b.valueAt(row, col);
			}
		}

		return new Matrix(matrix);
	}

	public void times(int scalar) {
		for (int row = 0; row < this.rows(); row++) {
			for (int col = 0; col < this.cols(); col++) {
				matrix[row][col] *= scalar;
			}
		}
	} 

	public Matrix transpose() {
		double[][] matrix = new double[this.cols()][this.rows()];
		for (int row = 0; row < this.rows(); row++) {
			for (int col = 0; col < this.cols(); col++) {
				matrix[col][row] = (this.matrix[row][col]);
			}
		}
		
		return new Matrix(matrix);
	}

	public Matrix dot(Matrix b) {
		if (this.cols() != b.rows()) {
			System.err.println("Matrix.dot(Matrix b) Exception:\nDimensions do not match. (" + this.cols() + " is not " + this.rows() + ")");
			System.exit(1);
		}

		int rows = this.rows();
		int cols = b.cols();
		double[][] matrix = new double[rows][cols];

		for (int row = 0; row < rows; row++) {
			for (int col = 0; col < cols; col++) {
				for (int len = 0; len < this.cols(); len++) {
					matrix[row][col] += this.valueAt(row, len) * b.valueAt(len, col);
				}
			}
		} 		

		return new Matrix(matrix);
	}
}
