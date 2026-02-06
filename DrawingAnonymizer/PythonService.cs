using Newtonsoft.Json;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace DrawingAnonymizer
{
    public class PythonResult
    {
        public bool Success { get; set; }
        public string Output { get; set; }
        public int Pages { get; set; }
    }

    public class PythonService
    {
        private readonly string pythonExe = "python";
        private readonly string scriptPath;
        public PythonService()
        {
            scriptPath = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "..\\..\\..\\python_backend\\anonymize.py"); //путь к скрипту питона
            scriptPath = Path.GetFullPath(scriptPath);
        }

        public async Task<PythonResult> RunAsync(string inputPdf, string outputPdf)
        {
            var psi = new ProcessStartInfo
            {
                FileName = pythonExe,
                Arguments = $"\"{scriptPath}\" \"{inputPdf}\" \"{outputPdf}\"",
                RedirectStandardOutput = true,
                UseShellExecute = false,
                CreateNoWindow = true
            };

            var process = Process.Start(psi);
            var json = await process.StandardOutput.ReadToEndAsync();
            await Task.Run(() =>  process.WaitForExit());

            return JsonConvert.DeserializeObject<PythonResult>(json);
        }
    }
}
