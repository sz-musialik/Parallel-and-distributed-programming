using System;
using System.Diagnostics;
using System.Threading;
using System.Threading.Tasks;
 
namespace lab1
{
    internal class Program
    {
        // parametry
        static int rozmiarZrodla = 2000;
        const int pojemnoscPojazdu = 200;
        const int czasPozyskania = 10;
        const int czasRozladowania = 10;
        const int czasPrzejazdu = 10000;
 
        const int liczbaGornikow = 5;
        static int totalStored = 0;
 
 
        static int sharedResource = 0;
        static SemaphoreSlim mineSemaphore = new SemaphoreSlim(2, 2);
        static SemaphoreSlim warehouseSemaphore = new SemaphoreSlim(1, 1);
 
        // blokady dostepu
        static object mineLock = new object();
        static object consoleLock = new object();
 
        static void Main(string[] args)
        {
            Console.WriteLine("Stan zloza: " + rozmiarZrodla + " jednostek wegla");
            Console.WriteLine("Stan magazynu: " + totalStored + " jednostek wegla\n");
            for (int j = 0; j < liczbaGornikow; j++)
            {
                Console.WriteLine("Gornik " + (j + 1) + ": ...");
            }
            Stopwatch stopwatch = Stopwatch.StartNew();
            Task[] tasks = new Task[liczbaGornikow];
 
            for (int i = 0; i < liczbaGornikow; i++)
            {
                int idGornika = i + 1;
                tasks[i] = Task.Run(() => Miner(idGornika));
            }
            Task.WaitAll(tasks);
 
            stopwatch.Stop();
        }
 
        static void Miner(int idGornika)
        {
            while (true)
            {
                int cargo = 0;
                string message;
 
                mineSemaphore.Wait();
                Thread.Sleep(100);
 
                try
                {
                    lock (mineLock)
                    {
                        if (rozmiarZrodla <= 0)
                        {
                            return;
                        }
 
                        cargo = Math.Min(pojemnoscPojazdu, rozmiarZrodla);
                        rozmiarZrodla -= cargo;
                    }
 
 
                    message = "Wydobywa wegiel";
                    Log(message, idGornika);
 
                    Thread.Sleep(cargo * czasPozyskania);
                }
                finally
                {
                    mineSemaphore.Release();
                }
 
                // transport
                message = "Transportuje do magazynu";
                Log(message, idGornika);
                Thread.Sleep(czasPrzejazdu);
                warehouseSemaphore.Wait();
 
                try
                {
                    message = "Rozładowuje węgiel";
                    Log(message, idGornika);
                    Thread.Sleep(cargo * czasRozladowania);
 
                    lock (mineLock)
                    {
                        totalStored += cargo;
                    }
                }
                finally
                {
                    warehouseSemaphore.Release();
                }
            }
 
            static void Log(string message, int idGornika)
            {
                lock (consoleLock)
                {
                    Console.SetCursorPosition(0, 0);
                    Console.WriteLine("Stan zloza: " + rozmiarZrodla + " jednostek wegla".PadRight(50));
                    Console.WriteLine("Stan magazynu: " + totalStored + " jednostek wegla".PadRight(50));
                    Console.WriteLine();
 
                    Console.SetCursorPosition(10, idGornika + 2);
                    Console.Write(message.PadRight(50));
                }
            }
        }
    }
}
