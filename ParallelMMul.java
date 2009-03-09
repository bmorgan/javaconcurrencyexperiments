import java.io.BufferedReader;
import java.io.DataInputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.Reader;
import java.io.StreamTokenizer;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.Semaphore;


public class ParallelMMul
{
	/*arg is name of a text file with first entry being the size of the matrix 'n', assuming a square matrix.
	 * 2*'n'*'n' floating point numbers should follow (multiply 2 n*n row major matrices together).
	 * multiply A x B = C is assumed to happen as follows.  A11 = dot product of first row of a and first columb of b.*/
	
	//goal of SM algorithm is to keep threads from blocking other threads due to shared resource (e.g. memory) contention.
	//therefore we try to maximize cache hits for each core.
	//simple way (though maybe not quite the best) to do this is by using sum matrices.  
	//	this allows cache to always hit on sub matrix A
	//	submatrix B will always miss, but something has to miss.
	//	if summartices are aligned to memory boundaries, then writing to c should occur in each local cache, 
	//	and there will be no contention for main memory
	//
	//  each thread computes a batch of submatrices of C.
	//	sub matrices of C are computed using rule SubMatrix(x,y) = MatrixDotProduct of row of SubMatrices of A and column of SM of B.
	//  ideally a submatrix should be small enough that 3 submatrices can fit in the local cache.
	//  then a submatrix of B gets evicted after each multiply.
	
	//	main thread dispatches threads and blocks on a semaphore var initialized to -(number of threads).
	//  this minimizes context switching (hopefully).
	
	//NOTE that synchronization is very simple for SM approach.  
	// the only synchronization is at termination of all worker threads.
	// since it is termination point, main doesn't need to notify threads after they have synced.
	// threads are completely oblivous to other threads.  there are absolutely no shared resources.
	// most parallel algorithms are much worse.
	
	//many additional optimizations could be done (especially if on NUMA architecture) like word alignment, NIO.., loopUnrolling etc.  
	
	static int DEFAULT_PREFERRED_SUB = 32;
	
	private static PartitionedMatrix _matrixA;
	private static PartitionedMatrix _matrixB;
	
	public static void main(String[] args)
	{
		
		try
		{		
			int size = 512;
			doTiming(size, 512, true);
			for(int subSize = 4; subSize <= size; subSize*=2)
			{
				System.out.println("speedup factor of "+((double)doTiming(size, subSize, false)/(double)doTiming(size, subSize, true))+" x");
			}
		}
		catch(Exception e)
		{
			e.printStackTrace();
			System.exit(-1);
		}
	}
	
	private static final long doTiming(int sqDim, int subSqDim, boolean parallel) throws Exception
	{
		_matrixA = new PartitionedMatrix(subSqDim,sqDim,sqDim,true,true);
		_matrixB = new PartitionedMatrix(subSqDim, sqDim,sqDim,true,true);
		long time = System.currentTimeMillis();
		if(parallel)
		{
			PartitionedMatrix.mulParallel(_matrixA, _matrixB,2);
		}
		else
		{
			PartitionedMatrix.mulParallel(_matrixA, _matrixB,1);
		}
		time = System.currentTimeMillis() - time;
		String type = parallel ? "parallel": "serial";
		System.out.print(String.format("%s %2$d x %2$d(%3$d) MatrixMul time = ", type, sqDim, subSqDim));
		System.out.println(Long.toString(time));
		return time;
	}
	
	private static final void readFromFile(String fileName, boolean  print) throws Exception
	{
		FileInputStream is = new FileInputStream(fileName);
		int xdimA, ydimA, xdimB, ydimB;
		StreamTokenizer t = new StreamTokenizer(new InputStreamReader(is, "US-ASCII"));
		
		t.nextToken();
		xdimA = (int)t.nval;
		t.nextToken();
		ydimA = (int)t.nval;
		_matrixA = new PartitionedMatrix(DEFAULT_PREFERRED_SUB,t, xdimA, ydimA);
		
		t.nextToken();
		xdimB = (int)t.nval;
		t.nextToken();
		ydimB = (int)t.nval;
		_matrixB = new PartitionedMatrix(DEFAULT_PREFERRED_SUB,t, xdimB, ydimB);
		
		System.out.println("finished reading A");
		if(print)_matrixA.printMatrix();
		System.out.flush();
		System.out.println("finished reading B");
		if(print)_matrixB.printMatrix();
		System.out.flush();
		is.close();
	}
	
	private static final class PartitionedMatrix
	{
		static int DEFAULT_THREADS = 2;
		
		public int _preferredSubDimx = DEFAULT_PREFERRED_SUB;	//preferred size of each submatrix.  
		public int _preferredSubDimy = DEFAULT_PREFERRED_SUB;
		/** 64*64*8 = 32KB * 3 matrices = 96KB which fits nicely in wolfdale's 128KB L1 cache.*/
	
		Semaphore _semaphore = new Semaphore(DEFAULT_THREADS);
		
		int _xDim;//size of full matrix.
		int _yDim;
		int _xSubDim;//number of submatrices
		int _ySubDim;
		
		SubMatrix[] _subMatrices;
		
		public PartitionedMatrix(int subSize, int xDim, int yDim, boolean allocateData)
		{this(subSize, xDim,yDim,allocateData,false);}
		
		public PartitionedMatrix(int subSize, StreamTokenizer t, int xDim, int yDim) throws Exception
		{
			this(subSize, xDim, yDim, true);
			readFromFile(t);
		}
		
		public PartitionedMatrix(int subSize, int xDim, int yDim, boolean allocateData, boolean randomData)
		{
			_preferredSubDimx = subSize; _preferredSubDimy = subSize;
			_xDim = xDim;
			_yDim = yDim;
			//allocate submatrices
			_xSubDim = (int)Math.ceil((float)xDim/_preferredSubDimx);
			_ySubDim = (int)Math.ceil((float)yDim/_preferredSubDimy);
			_subMatrices = new SubMatrix[_xSubDim * _ySubDim];
			
			if(!allocateData) return;
			
			for(int y = 0; y<_ySubDim; y++)
			{
				for(int x = 0; x < _xSubDim; x++)
				{
					int xsize = Math.min(_preferredSubDimx, xDim - x*_preferredSubDimx);
					int ysize = Math.min(_preferredSubDimy, yDim - y*_preferredSubDimy);
					double[] data = new double[xsize*ysize];
					_subMatrices[_xSubDim * y + x] = randomData ? 
							new SubMatrix(System.nanoTime(),x,y,xsize,ysize):
							new SubMatrix(x,y,xsize,ysize, data);
				}
			}
		}
		
		public final SubMatrix getSubMatrix(int x, int y)
		{
			return _subMatrices[y*_xSubDim + x];
		}
		
		public static PartitionedMatrix mulParallel(PartitionedMatrix a, PartitionedMatrix b, int threads)
		{
//			TODO fix sub size.  need to define it in 2 dims.
			PartitionedMatrix rval = new PartitionedMatrix(a._preferredSubDimx, b._xDim, a._yDim, true);
			Semaphore semaphore = new Semaphore(-threads + 1, false);	//workers will release and main will try to acquire.
			BatchWorker[] workers = new BatchWorker[threads];
			ArrayList<Job>[] batches = new ArrayList[threads];
			
			//initialize batch lists.
			for(int i = 0; i<threads; i++) 
				batches[i] = new ArrayList<Job>(rval._subMatrices.length/threads);
			//populate batches with jobs (round robin).
			int threadIdx = 0;
			for (int y = 0; y < rval._ySubDim; y++)
			{
				for (int x = 0; x < rval._xSubDim; x++)
				{
					Job job = new Job(a, b, y, x, rval.getSubMatrix(x,y));
					batches[threadIdx].add(job);
					threadIdx = (threadIdx + 1) % threads;
				}
			}
			//execute each batch on a new thread.
			for(int i = 0; i< threads; i++)
			{
				workers[i] = new BatchWorker(batches[i],semaphore);
				Thread t = new Thread(workers[i]);
				t.setPriority(6);
				t.start();
			}
			
			semaphore.acquireUninterruptibly();//block till all are done.
			return rval;
		}
		

		
		/** Job object used to represent the job that a single thread would perform.  
		 * Now, threadds will execute many jobs.
		 * Basically, a "Job" simply computes a SubMatrix of C in A x B = C
		 * using SubMatrices to do the MMul allows for optimal cache hits. */
		private static class Job
		{
			public final int _aRow, _bCol;
			public final SubMatrix _cSub;
			public final PartitionedMatrix _a, _b;
			public Job(PartitionedMatrix a, PartitionedMatrix b, int arow, int bcol, SubMatrix cSub)
			{
				_a = a; _b = b;
				assert a._xDim == b._yDim;
				_aRow = arow; _bCol = bcol;
				_cSub = cSub;//no guarentees that this ref is stable, but that's why method it private.
			}
			public void doJob()
			{
				for(int i = 0; i<_a._xSubDim; i++)
				{
					_cSub.mulAndAdd(_a.getSubMatrix(i, _aRow), _b.getSubMatrix(_bCol, i));
				}
			}
		}
		/** the Runnable that executes a batch of Jobs for parallel MMul */
		private static class BatchWorker implements Runnable
		{
			public volatile boolean done = false;
			private final Semaphore _sem;
			List<Job> _jobs;//jobs to be executed serially
			
			public BatchWorker(List<Job> jobs, Semaphore semaphore)
			{
				_jobs = jobs;
				_sem = semaphore;
			}
			
			public void run()
			{
				//in java 5 i should be guarenteed that final fields are coherent.
				//so each thread should be able to read fields of Job.
				for(Job j : _jobs)
				{
					j.doJob();
				}
				done = true;
				_sem.release();
			}
		}
		
//		public static PartitionedMatrix mulInPlaceParallel(PartitionedMatrix a, PartitionedMatrix b)
//		{
////			TODO fix sub size.  need to define it in 2 dims.
//			//PartitionedMatrix rval = new PartitionedMatrix(a._preferredSubDimx, b._xDim, a._yDim, true);
//			int threads = a._ySubDim * b._xSubDim;
//			Semaphore semaphore = new Semaphore(-threads + 1, false);	//workers will release and main will try to acquire.
//			InPlaceWorker[] workers = new InPlaceWorker[threads];
//			int threadIdx = 0;
//			for (int y = 0; y < a._ySubDim; y++)
//			{
//				for (int x = 0; x < b._xSubDim; x++)
//				{
//					InPlaceJob job = new InPlaceJob(a, b, y, x);
//					workers[threadIdx] = new InPlaceWorker(job, semaphore);
//					threadIdx++;
//				}
//			}
//			for(int i = 0; i< threads; i++)
//			{
//				Thread t = new Thread(workers[i]);
//				t.setPriority(6);
//				t.start();
//			}
//			
//			semaphore.acquireUninterruptibly();//block till all are done.
//			return b;
//		}
//		private static class InPlaceJob
//		{
//			public final int _aRow, _bCol;
//			public final PartitionedMatrix _a, _b;
//			public InPlaceJob(PartitionedMatrix a, PartitionedMatrix b, int arow, int bcol)
//			{
//				_a = a; _b = b;
//				assert a._yDim == b._yDim;
//				_aRow = arow; _bCol = bcol;
//				SubMatrix rSub = _b.getSubMatrix(_bCol, _aRow);
//				rSub._latch = new CountDownLatch(b._ySubDim);//sub in b
//			}
//			public void doJob() throws InterruptedException
//			{
//				SubMatrix rSub = _b.getSubMatrix(_bCol, _aRow);//sub in b
//				int newx = _b.getSubMatrix(_bCol, _aRow)._xDim;
//				int newy = _a.getSubMatrix(_bCol, _aRow)._yDim;
//				SubMatrix c = new SubMatrix(_bCol, _aRow, newx, newy, new double[newx*newy]);
//				for(int i = 0; i<_a._xSubDim; i++)
//				{
//					c.mulAndAdd(_a.getSubMatrix(i, _aRow), _b.getSubMatrix(_bCol, i));
//					_b.getSubMatrix(_bCol, i)._latch.countDown();//this job will never read this sub matrix again, so release it.
//				}
//				//wait for other threads to finish reading the sub in b where i will accumulate my result.
//				rSub._latch.await();
//				rSub._data = c._data;
//			}
//		}
//		
//		/** the Runnable that executes a batch of Jobs for parallel MMul */
//		private static class InPlaceWorker implements Runnable
//		{
//			public volatile boolean done = false;
//			InPlaceJob _job;
//			private final Semaphore _sem;
//			public InPlaceWorker(InPlaceJob job, Semaphore semaphore)
//			{
//				_sem = semaphore;
//				_job = job;
//			}
//			
//			public void run()
//			{
//				//in java 5 i should be guarenteed that final fields are coherent.
//				//so each thread should be able to read fields of Job.
//				while(true){
//					try{_job.doJob();break;
//					}catch(InterruptedException e){e.printStackTrace();}
//				}
//				done = true;
//				_sem.release();
//			}
//		}
		
		public void printMatrix()
		{
			SubMatrix.Printer[] printers = new SubMatrix.Printer[_xSubDim];
			for(int currSubY = 0; currSubY < _ySubDim; currSubY++)
			{
				//populate the printers for this row of submatrices.
				for(int currSubX = 0; currSubX < _xSubDim; currSubX++)
				{
					SubMatrix sm = _subMatrices[_xSubDim*currSubY + currSubX];
					printers[currSubX] = sm.getPrinter(); 
				}
				//for each row in the submatrices in this row of submatrices.
				//print a row in each submatrix.
				for(int subRow = 0; subRow< _subMatrices[_xSubDim * currSubY]._yDim; subRow++)
				{
					for(int currSubX = 0; currSubX < _xSubDim; currSubX++)
					{
						printers[currSubX].printRow();
						System.out.print("|");
						SubMatrix sm = _subMatrices[_xSubDim*currSubY + currSubX];
					}
					System.out.println();
				}
				System.out.println("---------------------------------------------------------");
			}
		}
		public void readFromFile(StreamTokenizer t) throws IOException
		{
//			read row by row into sub matrices.
			for(int currSubY = 0; currSubY < _ySubDim; currSubY++)
			{
				//for each row in the submatrices in this row of submatrices.
				//add one row of data to each submatrix.
				for(int subRow = 0; subRow< _subMatrices[_xSubDim * currSubY]._yDim; subRow++)
				{
					//add one row per submatrix.
					for(int currSubX = 0; currSubX < _xSubDim; currSubX++)
					{
						SubMatrix sm = _subMatrices[_xSubDim*currSubY + currSubX];
						int offset = sm._xDim * subRow;
						for(int elem = 0; elem < sm._xDim; elem++)
						{
							t.nextToken();
							while(t.ttype != StreamTokenizer.TT_NUMBER) 
								t.nextToken();
							double val = t.nval;
							sm._data[offset+elem] = val;
						}
					}
				}
			}
		}
	}
	
	private static class SubMatrix
	{
		int _xAddr;
		int _yAddr;
		int _xDim;
		int _yDim;
		double[] _data;
		CountDownLatch _latch; //used when doing a MMul in place A x B -> B
		
		public SubMatrix(long seed, int xAddr, int yAddr, int xDim, int yDim)
		{
			this(xAddr,yAddr,xDim,yDim,null);
			Random r = new Random(seed);
			_data = new double[xDim*yDim];
			for(int i = 0; i< _data.length; i++)
			{
				_data[i] = r.nextDouble();
			}
		}
		
		public SubMatrix(int xAddr, int yAddr, int xDim, int yDim, double[] data)
		{
			//data should be row major.
			assert (xDim * yDim) == data.length;
		
			_xAddr = xAddr;
			_yAddr = yAddr;
			_xDim = xDim;
			_yDim = yDim;
			_data = data;
		}
		
		public final void setElem(int x, int y, double val)
		{
			_data[y*_xDim + x] = val;
		}
		
		public final double getElem(int x, int y)
		{
			return _data[y*_xDim + x];
		}
		
		public Printer getPrinter()
		{
			return new Printer();
		}
		
		/** single threaded matrix multiply and add.  multiplies a*b then adds result to this matrix.*/
		public SubMatrix mulAndAdd(SubMatrix a, SubMatrix b)
		{
			assert (_yDim == a._yDim) && (_xDim == b._xDim); 
			for(int y = 0; y<_yDim; y++)
			{
				for(int x = 0; x<_xDim; x++)
				{
					//compute a subMatrix in rval
					for(int i = 0; i<a._xDim; i++)
					{
						//multiply and add row of SM of a and column of SM of b
						double elem = getElem(x,y);
						elem += a.getElem(i, y) * b.getElem(x, i);
						setElem(x, y, elem);
					}
				}
			}
			return this;
		}
		
		public void clear(){Arrays.fill(_data, 0.0);}
		
		/** single threaded matrix multiply and add.  multiplies a*b then adds result to this matrix.*/
		public SubMatrix Add(SubMatrix m)
		{
			assert _xDim == m._xDim && _yDim == m._yDim; 
			for(int y = 0; y<_yDim; y++)
			{
				for(int x = 0; x<_xDim; x++)
				{
					setElem(x, y, getElem(x, y) + m.getElem(x, y));
				}
			}
			return this;
		}
		
		/** single threaded matrix multiply and add.  multiplies a*b then adds result to this matrix.*/
		public static SubMatrix mul(SubMatrix a, SubMatrix b)
		{
			double[] data = new double[b._xDim*a._yDim];
			SubMatrix rval = new SubMatrix(a._yAddr, b._xAddr, b._xDim, a._yDim, data);
			rval.mulAndAdd(a, b);//rval is initialized to 0, so this is same as regular multiply;
			return rval;
		}
		
		public class Printer
		{
			int currentRow = 0;
			
			public void printRow()
			{
				StringBuilder sb = new StringBuilder(_xDim * 11);
				for(int i = 0; i<_xDim; i++)
				{
					double num = _data[currentRow*_xDim+i];
					sb.append(String.format("%1$ 9.4g ", num));
				}
				System.out.print(sb.toString());
				currentRow++;
			}
		}
		
	}
}
