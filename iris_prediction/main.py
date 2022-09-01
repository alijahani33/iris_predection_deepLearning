from Iris import iris


sepal_length = float(input("Sepal Length: "))
sepal_width  = float(input("Sepal Width: "))
petal_length = float(input("Petal Length: "))
petal_width  = float(input("Petal Width: "))

ir = iris.Iris()
predition = ir.predict(sepal_length,sepal_width,petal_length,petal_width)
print("Iris Type: "+predition)

