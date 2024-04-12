#Franco Gomez Kathie Malti CID Hands-on 2
import math

class Dataset:
    def __init__(self):
        self.x = [1, 2, 3, 4, 5, 6, 7, 8, 9] #Datos de X
        self.y = [5, 10, 15, 20, 25, 30, 35, 40, 45] #Datos de y
        self.x_values = [1, 2, 3, 4, 5, 6, 7, 8, 9] #Datos a predecir
        
        
class SLRcalculator:
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
       multiplied_sum = 9 *  xy_sum
       return multiplied_sum
    
    def numerator_r(self):
        return self.n_sumXY() - self.sumXsumY()

    def square_X(self):
        square_x = [x ** 2 for x in self.dataset.x]
        return sum(square_x) # X*X
    
    def square_Y(self):
        square_y = [y ** 2 for y in self.dataset.y]
        return sum(square_y)
    
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

    def predecir_y(self):
        return [168 + 23 * x for x in self.dataset.x_values] # para predecir los valores de Y en base a los de X
    
    def b0b1(self):
        n = len(self.dataset.x)
        X = self.sum_X()
        Y = self.sum_Y()
        X2 = self.square_X()
        XY = self.sum_XY()

        B0 = (Y * X2 - X * XY) / (n * X2 - X ** 2)
        B1 = (n * XY - X * Y) / (n * X2 - X ** 2)

        return B0, B1


def main():
    dataset_obj = Dataset()
    calculator = SLRcalculator(dataset_obj)
    predecir_Y = calculator.predecir_y()
    correlation, determination = calculator.coefficients()
    B0, B1 = calculator.b0b1()
    print("----------------------------------------------------------------------------------")
    print("----------------------------------------------------------------------------------")
    print("LINEAR REGRESSION EQUATION ( y = a + bx): ")
    print("----------------------------------------------------------------------------------\n")
    print("Sales = 168 + 23 Advertising \n")
    print("----------------------------------------------------------------------------------")
    print("Given X predict Y:")
    print("----------------------------------------------------------------------------------\n")
    print("Original Advertising (X values):\n", dataset_obj.x, '\n\n')
    print("Expected Sales (Original Y values):\n", dataset_obj.y, '\n\n')
    print("Predicted Sales (predicted Y values):\n", predecir_Y, '\n\n')
    print("----------------------------------------------------------------------------------")
    print("Correlation and determination coefficients ")
    print("----------------------------------------------------------------------------------")
    print("Correlation coefficient: ",correlation, '\n')
    print("Determination coefficient: ", determination)
    print("----------------------------------------------------------------------------------")
    print("----------------------------------------------------------------------------------")
    print("B0: ", B0, '\n')
    print("B1: ", B1)
    print("----------------------------------------------------------------------------------")
    print("----------------------------------------------------------------------------------\n")

if __name__ == "__main__":
    main()