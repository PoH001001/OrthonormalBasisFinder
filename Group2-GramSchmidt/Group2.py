import numpy as np #Library that contains the function needed for scientific computing of N-Dimensional Array
import sympy as sp #Library that helps in making sure that the output is readable
from flask import Flask, request, jsonify,render_template
import Store


app = Flask(__name__)

@app.route("/", methods=["GET"])
def home():
    return  render_template("index.html")

#LI_tester functions helps determine if the given matrix is Linearly
#Independent which is crucial in finding the Orthonormal Basis
def LI_tester(input):
    size = input.shape #Determines the Shape of the matrix

    #Compares the rows and columns
    if size[1] < size[0]:
        return False
    elif size[0] == size[1]:
        #Checks for the singularity of the matrix
        determinant = int(np.linalg.det(input))
        if determinant == 0:
            return False

        return True
    else:
        #Checks for any free variable
        pivot = np.linalg.matrix_rank(input)
        if pivot < size[0]:
            return False

    return True
    
#Sum_Projection functions helps in adding all the projection
# of a given matrix from the first vector up to the index of
# the vector you're comparing it with
def Sum_Projection(input, size, index):
    #Used to store the result of the summation
    result_matrix = np.zeros(size[1])

    #Used in computing the projection of a specific vector and getting their sum
    for i in range(index):
        norm = np.linalg.norm(input[i])
        result_matrix += ((input[index] @ input[i])/ norm**2) * input[i] 

    return result_matrix

#Normalization function helps in normalizing the given matrix
def Normalization(input, size):
    #Used to store the result of the normalization
    result_matrix = np.empty(size)

    #Used in getting the normalized vector of a given vector
    for i in range(size[0]):
        result_matrix[i] = input[i]/np.linalg.norm(input[i])
    
    return result_matrix

#Gram_Schmidt function is responsible for the overall execution of the Gram Schmidt process.
def Gram_Schmidt(input, size):
    #Used to store the result of the process
    result_matrix = np.empty(size)

    #Iteration for the whole matrix
    for i in range(size[0]):
        #Used to store the first vector w1 = v1
        if i == 0:
            result_matrix[i] = input[i]
        #Used in getting the orthogonal vectors of each vector
        else:
            result_matrix[i] = input[i] - Sum_Projection(result_matrix, size, i)

    #Used in normalizing the result of the iteration
    result_matrix = Normalization(result_matrix, size)

    return result_matrix


@app.route("/Group2", methods=["POST"])
#Main function is the over all interface of the program, which consists of the input, process, and output
def main():
    result = []
    data = request.get_json()

    matrix = np.array(data["matrix"])

    matrix = matrix.T

    matrix_size = matrix.shape

    #Checking for the independence
    independence = LI_tester(matrix)
    if independence == True:
        #Function call for the Gram Schmidt process
            result_matrix = Gram_Schmidt(matrix, matrix_size)

        #Formatting the result so that it will be readable
            result_matrix[np.abs(result_matrix) < 1e-12] = 0
            end_matrix = sp.Matrix(result_matrix).applyfunc(sp.nsimplify)

        #Printing of the result
            result_matrix[np.abs(result_matrix) < 1e-12] = 0

            #Formating
            end_matrix = sp.Matrix(result_matrix).applyfunc(
                lambda x: sp.nsimplify(x, tolerance=1e-5, rational=False)
            )

        #convert matrix to string list
            formatted_result = []
            for i in range(end_matrix.rows):
                row_strings = [str(end_matrix[i, j]) for j in range(end_matrix.cols)]
                formatted_result.append(row_strings)

            return jsonify({"result": formatted_result})
    else:

            flag = 0

            return jsonify({"result": flag})

if __name__ == "__main__":
    app.run(debug=True)
