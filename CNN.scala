package pplAssignment

object CNN {
	def dotrow(matrix_1: List[Double], matrix_2: List[Double], sum: Double):Double = {
		if(matrix_1.isEmpty || matrix_2.isEmpty)
			sum
		else
			dotrow(matrix_1.tail, matrix_2.tail, sum + matrix_1.head*matrix_2.head)
	}

	def dothelper(matrix_1: List[List[Double]], matrix_2: List[List[Double]], sum: Double):Double = {
		if(matrix_1.isEmpty || matrix_2.isEmpty)
    		sum
    	else
    		dothelper(matrix_1.tail, matrix_2.tail, sum + dotrow(matrix_1.head, matrix_2.head, 0.0))
	}

    def dotProduct(matrix_1: List[List[Double]], matrix_2: List[List[Double]]):Double = {
    	dothelper(matrix_1, matrix_2, 0.0)
    }

    def maxpool(list: List[Double]):Double = {
    	maxhelper(list, 0)
    }

    def avgpool(list: List[Double]):Double = {
    	val noofelements: Int = lencol(list, 0)
    	val sumofelements: Double = avgpoolhelper(list, 0)
    	val avg: Double = sumofelements/noofelements
    	avg
    }

    def avgpoolhelper(list: List[Double], sum: Double): Double = list match {
    	case Nil => sum
    	case a::b => avgpoolhelper(b, sum+a)
    }

    def selectrowelements(row: List[Double], size_row: Int, res: List[Double]):List[Double] = row match {
    	case Nil => res
    	case a::b => {
    		if(size_row == 0)
    			res
    		else
    			selectrowelements(b, size_row-1, res ::: List(a))
    	}
    }

    def selectrow(row: List[Double], col: Int, size_row: Int):List[Double] = row match {
    	case Nil => List()
    	case a::b => {
    		if(col == 0)
    			selectrowelements(row, size_row, List())
    		else
    			selectrow(b, col-1, size_row)
    	}
    }

    def selectmatrix(matrix: List[List[Double]], size_row: Int, size_col: Int, row: Int, col: Int):List[List[Double]] = matrix match {
    	case Nil => List()
    	case a::b => {
    		if(row == 0)
    			selectmatrixelements(matrix, size_row, size_col, col, List())
    		else
    			selectmatrix(b, size_row, size_col, row-1, col)
    	}
    }

    def selectmatrixelements(matrix: List[List[Double]], size_row: Int, size_col: Int, col: Int, res: List[List[Double]]):List[List[Double]] = matrix match {
    	case Nil => res
    	case a::b => {
    		if(size_col == 0)
    			res
    		else
    			selectmatrixelements(b, size_row, size_col-1, col, res ::: List(selectrow(a, col, size_row)))
    	}
    }

    def compute(Image: List[List[Double]], Kernel: List[List[Double]], imageSize: List[Int], kernelSize: List[Int], row: Int, col: Int, res: List[List[Double]], list: List[Double]):List[List[Double]] = {
		if(row == imageSize.head - kernelSize.head + 1)
			res
		else if(col == imageSize.tail.head - kernelSize.tail.head + 1)
			compute(Image, Kernel, imageSize, kernelSize, row+1, 0, res ::: List(list), List())
		else
		{
			val dp: Double = dotProduct(selectmatrix(Image, kernelSize.tail.head, kernelSize.head, row, col), Kernel)
			compute(Image, Kernel, imageSize, kernelSize, row, col+1, res, list ::: List(dp))
		}	
    }

    def convolute(Image:List[List[Double]], Kernel:List[List[Double]], imageSize:List[Int], kernelSize:List[Int]):List[List[Double]] = {
    	compute(Image, Kernel, imageSize, kernelSize, 0, 0, List(), List())
    }

    def poolhelper(poolingFunc: List[Double] => Double, Image: List[List[Double]], M: Int, K:Int, row: Int, col: Int, res: List[Double]):List[Double] = {
		if(col == M)
			res
		else
		{
			val mat: List[Double] = mattolist(selectmatrix(Image, K, K, row, col), List())
			poolhelper(poolingFunc, Image, M, K, row, col+K, res ::: List(poolingFunc(mat)))
		}
    }

    def mattolist(matrix: List[List[Double]], res: List[Double]): List[Double] = matrix match {
    	case Nil => res
    	case a::b => mattolist(b, res ::: a)
    }

    def singlePooling(poolingFunc: List[Double] => Double, Image: List[List[Double]], K: Int): List[Double] = {
    	poolhelper(poolingFunc, Image, lencol(Image.head, 0), K, 0, 0, List())
    }

    def poolingLayerhelper(poolingFunc: List[Double] => Double, Image: List[List[Double]], R: Int, M:Int, K: Int, row: Int, res: List[List[Double]]): List[List[Double]] = {
    	if(row == R)
    		res
    	else
    	{
    		poolingLayerhelper(poolingFunc, Image, R, M, K, row+K, res ::: List(singlePooling(poolingFunc, selectmatrix(Image, M, K, row, 0), K)))
    	}
    }

    def poolingLayer(poolingFunc: List[Double] => Double, Image: List[List[Double]], K: Int): List[List[Double]] = {
    	poolingLayerhelper(poolingFunc, Image, lenrow(Image, 0), lencol(Image.head, 0), K, 0, List())
    }

    def apply(activationFunc: Double => Double, row: List[Double], res: List[Double]): List[Double] = row match {
    	case Nil => res
    	case a::b => apply(activationFunc, b, res ::: List(activationFunc(a)))
    }

    def activationLayer(activationFunc:Double => Double, Image:List[List[Double]]):List[List[Double]] = {
    	activationhelper(activationFunc, Image, List())
    }

    def activationhelper(activationFunc:Double => Double, Image:List[List[Double]], res: List[List[Double]]):List[List[Double]] = Image match {
    	case Nil => res
    	case x1::x2 => activationhelper(activationFunc, x2, res ::: List(apply(activationFunc, x1, List())))
    }

    def normalise(Image: List[List[Double]]):List[List[Int]] = {
    	val minvalue: Double = min(Image, Image.head.head)
    	val maxvalue: Double = max(Image, 0)
    	normaliseextra(Image, minvalue, maxvalue, List())
    }

    def normaliseextra(Image: List[List[Double]], minvalue: Double, maxvalue: Double, res: List[List[Int]]):List[List[Int]] = Image match {
    	case Nil => res
    	case a::b => normaliseextra(b, minvalue, maxvalue, res ::: List(normalisehelper(a, minvalue, maxvalue, List())))
    }

    def normalisehelper(row: List[Double], minvalue: Double, maxvalue: Double, res: List[Int]):List[Int] = row match {
    	case Nil => res
    	case a::b => normalisehelper(b, minvalue, maxvalue, res ::: List(op(a, minvalue, maxvalue)))
    }

    def op(x: Double, minvalue: Double, maxvalue: Double):Int = {
    	val result: Int = Math.round((((x - minvalue)*255.0)/(maxvalue - minvalue))).toInt
    	result
    }

    def min(Image: List[List[Double]], res: Double):Double = Image match {
    	case Nil => res
    	case a::b => {
    		val x: Double = minhelper(a, a.head)
    		if(res < x)
    			min(b, res)
    		else
    			min(b, x)
    	}
    }

    def minhelper(row: List[Double], minvalue: Double):Double = row match {
    	case Nil => minvalue
    	case a::b => {
    		if(a < minvalue)
    			minhelper(b, a)
    		else
    			minhelper(b, minvalue)
    	}
    }

    def max(Image: List[List[Double]], res: Double):Double = Image match {
    	case Nil => res
    	case a::b => {
    		val x: Double = maxhelper(a, 0)
    		if(res > x)
    			max(b, res)
    		else
    			max(b, x)
    	}
    }

    def maxhelper(row: List[Double], maxvalue: Double):Double = row match {
    	case Nil => maxvalue
    	case a::b => {
    		if(a > maxvalue)
    			maxhelper(b, a)
    		else
    			maxhelper(b, maxvalue)
    	}
    }

    def mixedLayer(Image: List[List[Double]], Kernel: List[List[Double]], imageSize: List[Int], kernelSize: List[Int], activationFunc: Double => Double, poolingFunc: List[Double] => Double, K: Int):List[List[Double]] = {
    	val one: List[List[Double]] = convolute(Image, Kernel, imageSize, kernelSize)
    	val two: List[List[Double]] = activationLayer(activationFunc, one)
    	val three: List[List[Double]] = poolingLayer(poolingFunc, two, K)
    	three
    }

    def relu(x: Double):Double = {
    	if(x>=0)
    		x
    	else
    		0
    }

    def leakyrelu(x: Double):Double = {
    	if(x>=0)
    		x
    	else
    		0.5*x
    }

    def mul(matrix: List[List[Double]], f: Double, res: List[List[Double]]):List[List[Double]] = matrix match {
    	case Nil => res
    	case a::b => mul(b, f, res ::: List(mulhelper(a, f, List())))
    }

    def mulhelper(row: List[Double], f: Double, res: List[Double]):List[Double] = row match {
    	case Nil => res
    	case a::b => mulhelper(b, f, res ::: List(a*f))
    }

    def add(matrix: List[List[Double]], f: Double, res: List[List[Double]]):List[List[Double]] = matrix match {
    	case Nil => res
    	case a::b => add(b, f, res ::: List(addhelper(a, f, List()))) 
    }

    def addhelper(row: List[Double], f:Double, res: List[Double]):List[Double] = row match {
    	case Nil => res
    	case a::b => addhelper(b, f, res ::: List(a+f)) 
    }

    def addmatrices(matrix1: List[List[Double]], matrix2: List[List[Double]], res: List[List[Double]]):List[List[Double]] = {
    	if(matrix1.isEmpty && matrix2.isEmpty)
    		res
    	else
    		addmatrices(matrix1.tail, matrix2.tail, res ::: List(addmatriceshelper(matrix1.head, matrix2.head, List())))
    }

    def addmatriceshelper(row1: List[Double], row2: List[Double], res: List[Double]):List[Double] = {
    	if(row1.isEmpty && row2.isEmpty)
    		res
    	else
    		addmatriceshelper(row1.tail, row2.tail, res ::: List(row1.head + row2.head))
    }

   	def lenrow(matrix: List[List[Double]], sum: Int):Int = matrix match {
   		case Nil => sum
   		case a::b => lenrow(b, sum+1)
   	}

   	def lencol(row: List[Double], sum: Int):Int = row match {
   		case Nil => sum
   		case a::b => lencol(b, sum+1)
   	}

    def assembly(Image: List[List[Double]], imageSize: List[Int], w1: Double, w2: Double, b: Double, Kernel1: List[List[Double]], kernelSize1: List[Int], Kernel2: List[List[Double]], kernelSize2: List[Int], Kernel3: List[List[Double]], kernelSize3: List[Int], Size: Int):List[List[Int]] = {
    	val temp1: List[List[Double]] = mixedLayer(Image, Kernel1, imageSize, kernelSize1, relu, avgpool, Size)
    	val temp2: List[List[Double]] = mixedLayer(Image, Kernel2, imageSize, kernelSize2, relu, avgpool, Size)
    	val temp3: List[List[Double]] = mul(temp1, w1, List())
    	val temp4: List[List[Double]] = mul(temp2, w2, List())
    	val temp5: List[List[Double]] = addmatrices(temp3, temp4, List())
    	val temp6: List[List[Double]] = add(temp5, b, List())
    	val temp7: List[List[Double]] = mixedLayer(temp6, Kernel3, List(lenrow(temp6, 0), lencol(temp6.head, 0)), kernelSize3, leakyrelu, maxpool, Size)
    	val temp8: List[List[Int]] = normalise(temp7)
    	temp8
    }
}