import json
import dill
import pandas as pd
import os
import glob
from datetime import datetime

#------------------------------------------------------------------------------------------
def main_test_from_folder(n_files=10, sort_by='name', ascending=True):
# ------------------------------------------------------------------------------------------
    """
    # Тестирование модели на json-файлах, примерах из папки 'examples'
    # В папке 'examples' - 100 файлов (50 - с target_action = 1 и 50 - с target_action = 0)
    # Результаты тестирования записываются в папку 'test_results' в текстовый файл вида 'test_results_YYYY-mm-dd_HH-MM-SS.txt'
    # Время тестирования на 100 файлах порядка 30 секунд

    :param n_files:     количество файлов для тестирования
    :param sort_by:     способ сортировки файлов (name - по имени; edot_date - по дате изменения
    :param ascending:   вид сортировки (True - по возрастанию; False - по убыванию)
    :return:
    """
    # ------------------------------------------------------------------------------------------
    # Загрузка модели
    # filename_model = 'models/model_prediction_sber_final_(JN).pkl'
    filename_model = 'models/model_prediction_sber.pkl'
    with open(filename_model, 'rb') as file:
        model = dill.load(file)

    print(model['metadata'])

    pipe_model = model['model']
    # pipe_model = set_pipeline_debug_mode(pipe, is_debug=True)

    # Получение списка файлов из папки examples
    # Получаем список файлов по маске 'example*.json'
    examples_dir = 'examples'
    files = glob.glob(os.path.join('examples', 'example*.json'))

    # Оставляем только имена файлов (без пути)
    files = [os.path.basename(f) for f in files]

    # Сортировка файлов
    if sort_by == 'name':
        files.sort(key=lambda x: x, reverse=not ascending)
    elif sort_by == 'date_edit':
        files.sort(key=lambda x: os.path.getmtime(os.path.join(examples_dir, x)), reverse=not ascending)

    # Ограничение количества файлов
    if n_files != -1:
        files = files[:n_files]

    # Создание файла для записи результатов
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    result_filename = f'test_results/test_results_{timestamp}.txt'

    print('-' * 50 + '\n')

    with open(result_filename, 'w', encoding='utf-8') as result_file:
        # Запись заголовка в файл результатов
        result_file.write("Test Results\n")
        result_file.write(f"Model: {filename_model}\n")
        result_file.write(f"Test date: {timestamp}\n")
        result_file.write(f"Files tested: {len(files)}\n")
        result_file.write(f"Sort by: {sort_by}, order: {'ascending' if ascending else 'descending'}\n")
        result_file.write('-' * 50 + '\n\n')

        # Тестирование для каждого файла
        for filename in files:
            filepath = os.path.join(examples_dir, filename)
            with open(filepath, 'r', encoding='utf-8') as file_load:
                record_info = json.load(file_load)

            df = pd.DataFrame.from_dict([record_info])
            # y = model['model'].predict(df)
            y = pipe_model.predict(df)
            pred_res = y[0]

            # Формирование строки результата
            result_str = (f'file: {filename}\t'
                          f'session_id: {record_info["session_id"]}\t'
                          f'prediction: {pred_res}\t'
                          f'target_event: {record_info["target_event_action"]}\n')

            # Вывод на экран и запись в файл
            print(result_str, end='')
            result_file.write(result_str)

    print('\nТестирование закончено!')
    print(f'Результаты сохранены в файл: {result_filename}')


# ------------------------------------------------------------------------------------------
if __name__ == '__main__':
    main_test_from_folder(n_files=100, sort_by='name', ascending=True)
    # main_test_from_folder(n_files=10, sort_by='name', ascending=True)
