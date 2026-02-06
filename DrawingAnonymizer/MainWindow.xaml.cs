using System;
using System.Diagnostics;
using System.IO;
using System.Threading.Tasks;
using System.Windows;

namespace MetMashAnonymizer
{
    public partial class MainWindow : Window
    {
        private string _pdfPath;

        public MainWindow()
        {
            InitializeComponent();
        }

        private void SelectPdfButton_Click(object sender, RoutedEventArgs e)
        {
            var dialog = new Microsoft.Win32.OpenFileDialog();
            dialog.Filter = "PDF файлы|*.pdf";
            if (dialog.ShowDialog() == true)
            {
                _pdfPath = dialog.FileName;
                //FilePathTextBlock.Text = $"Выбрано: {System.IO.Path.GetFileName(_pdfPath)}";
            }
        }

        private async void AnonymizeButton_Click(object sender, RoutedEventArgs e)
        {
            if (string.IsNullOrEmpty(_pdfPath) || !File.Exists(_pdfPath))
            {
                MessageBox.Show("Сначала выберите PDF!");
                return;
            }

            var pythonScript = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "python_backend", "anonymize.py");
            var outputFile = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.Desktop), "result_anonymized.pdf");

            if (!File.Exists(pythonScript))
            {
                MessageBox.Show($"Не найден anonymize.py!\nПроверьте папку:\n{pythonScript}");
                return;
            }

            try
            {
                var psi = new ProcessStartInfo
                {
                    FileName = "python",
                    Arguments = $"\"{pythonScript}\" \"{_pdfPath}\" \"{outputFile}\"",
                    UseShellExecute = false,
                    CreateNoWindow = true
                };

                var process = Process.Start(psi);
                await Task.Run(() => process.WaitForExit());

                if (File.Exists(outputFile))
                {
                    MessageBox.Show($"✅ Готово!\nФайл сохранён на рабочем столе:\n{outputFile}");
                }
                else
                {
                    MessageBox.Show("❌ Ошибка: файл не создан.");
                }
            }
            catch (Exception ex)
            {
                MessageBox.Show($"Ошибка запуска Python:\n{ex.Message}");
            }
        }
    }
}