import scala.util.Random
import scala.collection.optimizer._

package benchmarks {

import org.openjdk.jmh.annotations.Benchmark
import org.openjdk.jmh.annotations.Scope
import org.openjdk.jmh.annotations.Setup
import org.openjdk.jmh.annotations.State
import org.openjdk.jmh.annotations.BenchmarkMode
import org.openjdk.jmh.annotations.Mode
import org.openjdk.jmh.annotations.OutputTimeUnit

import java.util.concurrent.TimeUnit

class Ref(var num: Int = 0)

@OutputTimeUnit(TimeUnit.NANOSECONDS)
@BenchmarkMode(Array(Mode.AverageTime))
@State(Scope.Thread)
class ClashOfLambdas {

  var N : Int = _
  var v : Array[Long] = _
  var vHi : Array[Long] = _
  var vLo : Array[Long] = _
  var refs : Array[Ref] = _

  @Setup
  def prepare() : Unit = {
    N = 10000000
    v = (0 until N).map(i => i.toLong % 1000).toArray
    vHi = (0 until 1000000).map(i => i.toLong).toArray
    vLo = (0 until 10).map(i => i.toLong).toArray
    refs = (0 until N).map(i => new Ref(i)).toArray
  }

  ////////////////////////////////////////
  // Benchmarks with Views (non-strict) //
  ////////////////////////////////////////

  @Benchmark
  def sumBaseline () : Long = {
    var i=0
    var sum=0L
    while (i < v.length) {
      sum += v(i)
      i += 1
    }
    sum
  }

  @Benchmark
  def sumOfSquaresBaseline () : Long = {
    var i=0
    var sum=0L
    while (i < v.length) {
      sum += v(i) *  v(i)
      i += 1
    }
    sum
  }

  @Benchmark
  def sumOfSquaresEvenBaseline () : Long = {
    var i=0
    var sum=0L
    while (i < v.length) {
      if (v(i) % 2 == 0)
        sum += v(i) * v(i)
      i += 1
    }
    sum
  }

  @Benchmark
  def cartBaseline () : Long = {
    var d, dp=0
    var sum=0L
    while (d < vHi.length) {
      dp = 0
      while (dp < vLo.length) {
        sum += vHi(d) * vLo(dp)
        dp +=1
      }
      d += 1
    }
    sum
  }


  @Benchmark
  def refBaseline () : Int = {
    var i=0
    var count=0
    while (i < refs.length) {
      if (refs(i).num % 5 == 0 && refs(i).num % 7 == 0)
        count += 1
      i += 1
    }
    count
  }

  @Benchmark
  def sumSeq () : Long = {
    val sum : Long = v
      .view
      .sum
    sum
  }

  @Benchmark
  def sumPar () : Long = {
    val sum : Long = v
      .par
      .view
      .sum
    sum
  }

  @Benchmark
  def sumOfSquaresSeq () : Long = {
    val sum : Long = v
      .view
      .map(d => d * d)
      .sum
    sum
  }

  @Benchmark
  def sumOfSquaresPar () : Long = {
    val sum : Long = v
      .par
      .view
      .map(d => d * d)
      .sum
    sum
  }

  @Benchmark
  def cartSeq () : Long = {
    val sum : Long = vHi
      .view
      .flatMap(d => vLo.view.map (dp => dp * d))
      .sum
    sum
  }

  @Benchmark
  def cartPar () : Long = {
    val sum : Long = vHi
      .par
      .view
      .flatMap(d => vLo.view.map (dp => dp * d))
      .sum
    sum
  }

  @Benchmark
  def sumOfSquaresEvenSeq () : Long = {
    val res : Long = v
      .view
      .filter(x => x % 2 == 0)
      .map(x => x * x)
      .sum
    res
  }

  @Benchmark
  def sumOfSquaresEvenPar () : Long = {
    val res : Long = v
      .par
      .view
      .filter(x => x % 2 == 0)
      .map(x => x * x)
      .sum
    res
  }

  @Benchmark
  def sumSeqOpt () : Long = {
    optimize {
      val sum : Long = v
        .sum
      sum
    }
  }

  @Benchmark
  def sumOfSquaresSeqOpt () : Long = {
    optimize {
      val sum : Long = v
        .map(d => d * d)
        .sum
      sum
    }
  }

  @Benchmark
  def cartSeqOpt () : Long = {
    optimize {
      val sum : Long = vHi
        .flatMap(d => vLo.map (dp => dp * d))
        .sum
      sum
    }
  }

  @Benchmark
  def sumOfSquaresEvenSeqOpt () : Long = {
    optimize {
      val res : Long = v
        .filter(x => x % 2 == 0)
        .map(x => x * x)
        .sum
      res
    }
  }

  @Benchmark
  def sumParOpt () : Long = {
    import scala.collection.par._
    import Scheduler.Implicits.global

    val sum : Long = v
      .toPar
      .sum
    sum
  }

  @Benchmark
  def sumOfSquaresParOpt () : Long = {
    import scala.collection.par._
    import Scheduler.Implicits.global

    val sum : Long = v
      .toPar
      .map(d => d * d)
      .sum
    sum
  }

  @Benchmark
  def cartParOpt () : Long = {
    import scala.collection.par._
    import Scheduler.Implicits.global

    val sum : Long = vHi
      .toPar
      .flatMap(d => vLo.map (dp => dp * d))
      .sum
    sum
  }

  @Benchmark
  def sumOfSquaresEvenParOpt () : Long = {
    import scala.collection.par._
    import Scheduler.Implicits.global
    import scala.reflect.ClassTag // https://github.com/scala-blitz/scala-blitz/issues/34

    val res : Long = v
      .toPar
      .filter(x => x % 2 == 0)
      .map(x => x * x)
      .sum
    res
  }

  @Benchmark
  def refSeq () : Int = {
    val res : Int = refs
      .view
      .filter(_.num % 5 == 0)
      .filter(_.num % 7 == 0)
      .size
    res
  }

  @Benchmark
  def refPar () : Int = {
    val res : Int = refs
      .par
      .view
      .filter(_.num % 5 == 0)
      .filter(_.num % 7 == 0)
      .size
    res
  }

  @Benchmark
  def refSeqOpt () : Int = {
    optimize {
      val res : Int = refs
        .filter(_.num % 5 == 0)
        .filter(_.num % 7 == 0)
        .size
      res
    }
  }

  @Benchmark
  def refParOpt () : Int = {
    import scala.collection.par._
    import Scheduler.Implicits.global
    import scala.reflect.ClassTag // https://github.com/scala-blitz/scala-blitz/issues/34

    val res : Int = refs
      .toPar
      .filter(_.num % 5 == 0)
      .filter(_.num % 7 == 0)
      .seq.size // the `size` method is not defined on the `Par` wrapper
    res
  }

  ///////////////////////////////////////
  // Benchmarks without Views (strict) //
  ///////////////////////////////////////
  @Benchmark
  def sumSeq_Strict () : Long = {
    val sum : Long = v
      .sum
    sum
  }

  @Benchmark
  def sumPar_Strict () : Long = {
    val sum : Long = v
      .par
      .sum
    sum
  }

  @Benchmark
  def sumOfSquaresSeq_Strict () : Long = {
    val sum : Long = v
      .map(d => d * d)
      .sum
    sum
  }

  @Benchmark
  def sumOfSquaresPar_Strict () : Long = {
    val sum : Long = v
      .par
      .map(d => d * d)
      .sum
    sum
  }

  @Benchmark
  def sumOfSquaresEvenSeq_Strict () : Long = {
    val res : Long = v
      .filter(x => x % 2 == 0)
      .map(x => x * x)
      .sum
    res
  }

  @Benchmark
  def sumOfSquaresEvenPar_Strict () : Long = {
    val res : Long = v
      .par
      .filter(x => x % 2 == 0)
      .map(x => x * x)
      .sum
    res
  }

  @Benchmark
  def cartSeq_Strict () : Long = {
    val sum : Long = vHi
      .flatMap(d => vLo.map (dp => dp * d))
      .sum
    sum
  }

  @Benchmark
  def cartPar_Strict () : Long = {
    val sum : Long = vHi
      .par
      .flatMap(d => vLo.map (dp => dp * d))
      .sum
    sum
  }

  @Benchmark
  def refSeq_Strict () : Int = {
    val res : Int = refs
      .filter(_.num % 5 == 0)
      .filter(_.num % 7 == 0)
      .size
    res
  }

  @Benchmark
  def refPar_Strict () : Int = {
    val res : Int = refs
      .par
      .filter(_.num % 5 == 0)
      .filter(_.num % 7 == 0)
      .size
    res
  }
}
}