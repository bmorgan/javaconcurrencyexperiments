import java.io.BufferedWriter;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.OutputStream;
import java.io.OutputStreamWriter;
import java.util.Random;


public class MatrixMaker
{
	/**@param args two integer args.  x dim, and y dim. */
	public static void main(String[] args)
	{
		File f = new File("matrices.txt");
		try
		{
			int cols = Integer.parseInt(args[0]);
			int rows = Integer.parseInt(args[1]);
			FileOutputStream os = new FileOutputStream(f);
			BufferedWriter w = new BufferedWriter(new OutputStreamWriter(os, "US-ASCII"));
			w.write(Integer.toString(cols)+"\n");
			w.write(Integer.toString(rows)+"\n");
			doMatrix(cols, rows, w);
			w.write("\n");
			//swap
			int tmp = cols; cols = rows; rows = tmp;
			
			w.write(Integer.toString(cols)+"\n");
			w.write(Integer.toString(rows)+"\n");
			doMatrix(cols, rows, w);
			
			w.flush();
			w.close();
		}
		catch(Exception e)
		{
			e.printStackTrace();
		}
	}
	
	private static void doMatrix(int cols, int rows, BufferedWriter w) throws IOException
	{
		Random r = new Random(System.nanoTime()); 
		
		for(int y = 0; y<rows; y++)
		{
			for(int x = 0; x<cols; x++)
			{
				double val = r.nextDouble() * 2.0 - 1.0;
				String s = Double.toString(val);
				w.write(s);
				w.write(" ");
			}
		}
	}
}

