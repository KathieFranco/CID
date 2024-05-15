#Franco Gomez Kathie Malti

import math

class Dataset:
    def __init__(self):
        self.x = [108, 115, 106, 97, 95, 91, 97, 83, 83, 78, 54, 67, 56, 53, 61, 115, 81, 78, 30, 45, 99, 32, 25, 28, 90, 89] #batch size
        self.y = [95, 96, 95, 97, 93, 94, 95, 93, 92, 86, 73, 80, 65, 69, 77, 96, 87, 89, 60, 63, 95, 61, 55, 56, 94, 93] #machine learning
        self.x_predecir = [62, 67, 77, 83, 90, 95] # data set de datos para predecir Y

        self.n = len(self.x)

class linearRegression:
    def __init__(self, dataset):
        self.dataset = dataset  
    # matriz x lineal
    def x_linear(self):
        x_linear = [[1, x] for x in self.dataset.x]
        return x_linear 
    # transpuesta de x lineal
    def x_transpuesta(self, x_linear):
        x_transpuesta = [[row[i] for row in x_linear] for i in range(2)]
        return x_transpuesta
    # mutltiplicacion de la matriz transpuesta por x lineal
    def x_transpuesta_x_linear(self, x_linear):
        x_transpuesta_x_linear = [[sum(X_row[i] * X_row[j] for X_row in x_linear) for j in range(2)] for i in range(2)]
        return x_transpuesta_x_linear
    
    '''def predecir_y_linear(self):
        result = [168 + 23 * [1, x] for x in self.dataset.x] # para predecir los valores de Y en base a los de X
        return result'''
    # hacer predicciones lineales
    def predicciones(self, result):
        B0 = result[0]
        B1 = result[1]
        return [B0 + B1 * x for x in self.dataset.x_predecir]
        

    
class quadraticRegression:
    def __init__(self, dataset):
        self.dataset = dataset  

    def x_quadratic(self):
        x_quadratic = [[1, x, x**2] for x in self.dataset.x]
        return x_quadratic
    
    def xT_quadratic(self, x_quadratic):
        xT_quadratic = [[row[i] for row in x_quadratic] for i in range(3)] 
        return xT_quadratic
    
    def xT_quadratic_x_quadratic(self, x_quadratic):
        xT_quadratic_x_quadratic = [[sum(X_row[i] * X_row[j] for X_row in x_quadratic) for j in range(3)] for i in range(3)]
        return xT_quadratic_x_quadratic
    
    def predicciones_qa(self, result):
        B0 = result[0]
        B1 = result[1]
        B2 = result[2]
        return [B0 + B1 * x + B2 * x**2 for x in self.dataset.x_predecir]

class cubicRegression:
    def __init__(self, dataset):
        self.dataset = dataset  

    def x_cubic(self):
        x_cubic = [[1, x, x**2, x**3] for x in self.dataset.x]
        return x_cubic
    
    def xT_cubic(self, x_cubic):
        xT_cubic = [[row[i] for row in x_cubic] for i in range(4)] 
        return xT_cubic
    # x transpuesta * matriz cubica
    def xT_cubic_x_cubic(self, x_cubic):
        xT_cubic_x_cubic = [[sum(X_row[i] * X_row[j] for X_row in x_cubic) for j in range(4)] for i in range(4)]
        return xT_cubic_x_cubic
    
    def predicciones_cu(self, result):
        B0 = result[0]
        B1 = result[1]
        B2 = result[2]
        B3 = result[3]
        return [B0 + B1 * x + B2 * x**2 + B3 * x**3 for x in self.dataset.x_predecir]
    
class matrix_calculator:

    # Calculate the inverse of x_transpuesta_x_linear using Gauss-Jordan elimination
    def calculate_inverse(self, matrix):
        n = len(matrix)
        identity = [[1 if i == j else 0 for j in range(n)] for i in range(n)]
        augmented_matrix = [row + identity[i] for i, row in enumerate(matrix)]
        
        # Perform Gauss-Jordan elimination
        for i in range(n):
            pivot = augmented_matrix[i][i]
            for j in range(i+1, n):
                ratio = augmented_matrix[j][i] / pivot
                augmented_matrix[j] = [a - ratio*b for a, b in zip(augmented_matrix[j], augmented_matrix[i])]
        
        # Backward substitution
        for i in range(n-1, -1, -1):
            pivot = augmented_matrix[i][i]
            augmented_matrix[i] = [a / pivot for a in augmented_matrix[i]]
            for j in range(i-1, -1, -1):
                ratio = augmented_matrix[j][i]
                augmented_matrix[j] = [a - ratio*b for a, b in zip(augmented_matrix[j], augmented_matrix[i])]
        
        # Extract the inverse matrix
        inverse_matrix = [row[n:] for row in augmented_matrix]
        
        return inverse_matrix
    
    def x_transpuesta_y(self, y, x_transpuesta):
        x_transpuesta_y = [sum(a*b for a, b in zip(row_X_T, y)) for row_X_T in x_transpuesta]
        return x_transpuesta_y
    
    def coefficients(self, x_transpuesta_y, x_transpuesta_x): 
        inverse = self.calculate_inverse(x_transpuesta_x)
        result = [sum(a*b for a, b in zip(row_inv, x_transpuesta_y)) for row_inv in inverse]
        return result
    

class imprimir:

    def show_x(self, x):
        print("\nX =") # (X)
        for row in x:
            print(row)
    
    def show_xT(self, xT):
        print("\n(X^T) =") #(X^T)
        for row in xT:
            print(row)

    def show_xT_x(self, xT_x):
        print("\n(X^T * X) =") # (X^T*X)
        for row in xT_x:
            print(row)
    
    def show_inverse_xT_x(self, inverse_xT_x):
        print("\n(X^T*X)^-1 =") # (X^T*X)^-1
        for row in inverse_xT_x:
            print(row)
    
    def show_predictions(self, predecir_y):
        print("\nPredicciones:") # (X^T*X)
        for row in predecir_y:
                print("Y =", row)
# esta clase solo sirve para sacar el coeficiente de determinacion y correlacion lineal
class Calculator:
    def __init__(self, dataset):
        self.dataset = dataset  # inicializamos la variable 
    
    def sum_X(self):
        return sum(self.dataset.x) #suma de valores de X

    def sum_Y(self):
        return sum(self.dataset.y) #suma de valores de Y
    
    def multiplicacion_XY(self):
        return [x * y for x, y in zip(self.dataset.x, self.dataset.y)] #multiplicacion de X * Y

    def sum_XY(self):
        return sum(self.multiplicacion_XY()) # SumXY

    def sumXsumY(self):
        return self.sum_X() * self.sum_Y() #SumX * SumY

    def sumXsumX(self):
        return self.sum_X() * self.sum_X() # SumX * SumX

    def sumYsumY(self):
        return self.sum_Y() * self.sum_Y() # SumY * SumY

    def n_sumXY(self):
       xy_sum = self.sum_XY() # n * SumXY
       n = self.dataset.n
       multiplied_sum = n *  xy_sum
       return multiplied_sum
    
    def numerator_r(self):
        return self.n_sumXY() - self.sumXsumY()

    def square_X(self):
        square_x = [x ** 2 for x in self.dataset.x]
        return sum(square_x) # X*X  = x^2
    
    def square_Y(self):
        square_y = [y ** 2 for y in self.dataset.y]
        return sum(square_y) # y^2
    
    def coefficients(self):
        n = len(self.dataset.x)
        x2 = self.square_X()
        sumXsumX = self.sumXsumX()
        y2 = self.square_Y()
        sumYsumY = self.sumYsumY()

        numerator_r = self.n_sumXY() - self.sumXsumY()
        denominator_r = math.sqrt((n * x2 - (sumXsumX))*(n * y2 -(sumYsumY)))


        correlation_r = numerator_r / denominator_r
        determination_r2 = correlation_r ** 2

        return correlation_r, determination_r2
# clase en proceso para calcular la correlacion y determinacion cuadratica
class calcular_r2:
    def __init__(self, dataset):
        self.dataset = dataset  # inicializamos la variable 
    # Calcular promedio del data set X
    def promedio_X(self): 
        suma = sum(self.dataset.x)
        n = len(self.dataset.x)
        return suma/n

    # Calcular promedio del data set Y
    def promedio_Y(self): 
        suma = sum(self.dataset.y)
        n = len(self.dataset.y)
        return suma/n
    # calcular y'
    def y_prima(self, result):
        B0 = result[0]
        B1 = result[1]
        B2 = result[2]
        y_prim = [B0 + B1 * x + B2 * x**2 for x in self.dataset.x]
        return y_prim
    # (y'-y_promedio)^2
    def resta_Yprima_Ypromedio(self, promedio_Y, result):
        y_prim = self.y_prima(result)
        return [(y - promedio_Y) ** 2 for y in y_prim]
    # (y-y_promedio)^2
    def resta_Y_Ypromedio(self, promedio_Y):
        return [(y - promedio_Y) ** 2 for y in self.dataset.y]
    
def main():
   
    dataset_obj = Dataset()
    mostrar = imprimir()
    inverse = matrix_calculator()
    calculator = Calculator(dataset_obj)
    correlation, determination = calculator.coefficients()
    calculate_r2 = calcular_r2(dataset_obj)
    r2 = calculate_r2.promedio_X()
    
    # Empieza linear regression
    linear_regression = linearRegression(dataset_obj)
    x_linear = linear_regression.x_linear()
    xT_linear = linear_regression.x_transpuesta(x_linear)
    xT_x_linear = linear_regression.x_transpuesta_x_linear(x_linear)
    inverse_xT_x_linear = inverse.calculate_inverse(xT_x_linear)
    xT_y_linear = inverse.x_transpuesta_y(dataset_obj.y, xT_linear)
    result_linear = inverse.coefficients(xT_y_linear, xT_x_linear) # resultados de B0, B1
    predecir_y_lineal = linear_regression.predicciones(result_linear)
    
    # Comienza Quadratic regression
    quadratic_regresion = quadraticRegression(dataset_obj)
    x_quadratic = quadratic_regresion.x_quadratic()
    xT_quadratic = quadratic_regresion.xT_quadratic(x_quadratic)
    xT_x_quadratic = quadratic_regresion.xT_quadratic_x_quadratic(x_quadratic)
    inverse_xT_x_quadratic = inverse.calculate_inverse(xT_x_quadratic)
    xT_y_quadratic = inverse.x_transpuesta_y(dataset_obj.y, xT_quadratic)
    result_quadratic = inverse.coefficients(xT_y_quadratic, xT_x_quadratic) # resultados de B0, B1, B2
    predecir_y_cuadratica = quadratic_regresion.predicciones_qa(result_quadratic)
    # Comienza Cubic regression
    cubic_regression = cubicRegression(dataset_obj)
    x_cubic = cubic_regression.x_cubic()
    xT_cubic = cubic_regression.xT_cubic(x_cubic)
    xT_x_cubic = cubic_regression.xT_cubic_x_cubic(x_cubic)
    inverse_xT_x_cubic = inverse.calculate_inverse(xT_x_cubic)
    xT_y_cubic = inverse.x_transpuesta_y(dataset_obj.y, xT_cubic)
    result_cubic = inverse.coefficients(xT_y_cubic, xT_x_cubic) # resultados de B0, B1, B2 y B3
    predecir_y_cubica = cubic_regression.predicciones_cu(result_cubic)
    # pruebas de impresion para los coeficientes cuadraticos
    r2 = calculate_r2.promedio_X()
    promY = calculate_r2.promedio_Y()
    y_prim = calculate_r2.y_prima(result_quadratic)
    resta_yprim_yprom = calculate_r2.resta_Yprima_Ypromedio(promY, result_quadratic)


    print("\nLINEAR CALCULATIONS")
    # imrpimimos x lineal [1, x], [1, ...], [1, x]
    # mostrar.show_x(x_linear) # X
    # imprimimos la transversa de x [1, ..., 1], [x, ..., x]
    # mostrar.show_xT(xT_linear) #(X^T)
    # imprimimos la multiplicación de la transversa de X por X lineal
    # mostrar.show_xT_x(xT_x_linear) # (X^T*X)
    # imprimimos la inversa de xT * x lineal
    # mostrar.show_inverse_xT_x(inverse_xT_x_linear) # (X^T*X)^-1
    # imprimimos la inversa de x * y
    # print("X^T * y =", xT_y_linear)
    B0=result_linear[0]
    B1=result_linear[1]
    print("Y =", B0, '+', B1,'x')
    # print("Predicciones:\n", predecir_y, '\n\n')
    mostrar.show_predictions(predecir_y_lineal)
    print("\nCorrelation: ",correlation, '')
    print("Determination: ", determination)
    # print("B0 =",B0)
    # print("B1 =",B1)
    # Comienza Quadratic regression
    print("\nQUADRATIC CALCULATIONS")
    # mostrar.show_x(x_quadratic)
    # mostrar.show_xT(xT_quadratic)
    # mostrar.show_xT_x(xT_x_quadratic)
    # mostrar.show_inverse_xT_x(inverse_xT_x_quadratic)
    # print("X^T * y =", xT_y_quadratic)
    B0 = result_quadratic[0]
    B1 = result_quadratic[1]
    B2 = result_quadratic[2]
    print("Y =", B0, '+', B1,'x','+', B2,'x^2')
    mostrar.show_predictions(predecir_y_cuadratica)
    # print("Coefficients =", result_quadratic)
    # Empieza Cubic regression
    print("\nCUBIC CALCULATIONS")
    # mostrar.show_x(x_cubic)
    # mostrar.show_xT(xT_cubic)
    # mostrar.show_xT_x(xT_x_cubic)
    # mostrar.show_inverse_xT_x(inverse_xT_x_cubic)
    # print("X^T * y =", xT_y_cubic)
    B0 = result_cubic[0]
    B1 = result_cubic[1]
    B2 = result_cubic[2]
    B3 = result_cubic[3]
    print("Y =", B0, '+', B1,'x','+', B2,'x^2','+', B3,'x^3')
    mostrar.show_predictions(predecir_y_cubica)

    #print(promY)
    #print(y_prim)
    # print(resta_yprim_yprom)
   

if __name__ == "__main__":
    main()
