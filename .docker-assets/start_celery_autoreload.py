# coding: utf-8

# borrow from https://gist.github.com/chenjianjx/53d8c2317f6023dc2fa0

"""
A python script which starts celery worker and auto reload it when any code change happens.
I did this because Celery worker's "--autoreload" option seems not working for a lot of people.
"""

import time
from watchdog.observers import Observer  # pip install watchdog
from watchdog.events import PatternMatchingEventHandler
import psutil  # pip install psutil
import os
import subprocess

code_dir_to_monitor = '/app'
celery_working_dir = code_dir_to_monitor  # happen to be the same. It may be different on your machine
celery_cmdline = 'celery worker -A albedo -l INFO'.split(' ')


class MyHandler(PatternMatchingEventHandler):

    def on_any_event(self, event):
        # print('Detected change. event = {0}'.format(event))

        for proc in psutil.process_iter():
            proc_cmdline = self._get_proc_cmdline(proc)
            if not proc_cmdline or len(proc_cmdline) < len(celery_cmdline):
                continue

            is_celery_worker = 'python' in proc_cmdline[0].lower() \
                               and celery_cmdline[0] == proc_cmdline[1] \
                               and celery_cmdline[1] == proc_cmdline[2]

            if not is_celery_worker:
                continue

            proc.kill()
            # print('Just killed {0} on working dir {1}'.format(proc_cmdline, proc.cwd()))

        run_worker()

    def _get_proc_cmdline(self, proc):
        try:
            return proc.cmdline()
        except Exception as e:
            return []


def run_worker():
    # print('Ready to call {0} '.format(celery_cmdline))
    os.chdir(celery_working_dir)
    subprocess.Popen(celery_cmdline)
    # print('Done callling {0} '.format(celery_cmdline))


if __name__ == '__main__':
    run_worker()

    event_handler = MyHandler(patterns = ['*.py'])
    observer = Observer()
    observer.schedule(event_handler, code_dir_to_monitor, recursive=True)
    observer.start()
    # print('File change observer started')

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()
