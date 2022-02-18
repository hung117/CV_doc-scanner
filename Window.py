from asyncio.windows_events import NULL
from ctypes import util
from sys import warnoptions
import PySimpleGUI as sg
import cv_utils as utils
from tkinter import filedialog
import cv2
import os
sg.theme('DarkGreen4')


def WARN_POPUP(warn='placeHolder'):
    strWarn = 'Error: '+warn
    sg.popup('Warning', strWarn)


def SAVE_FILE(frame):
    # Image directory
    initDir = os.getcwd()
    initDir += '\\out'
# change current directory
    os.chdir(initDir)
# Using cv2.imwrite() method Saving the image
    cv2.imwrite('img.jpg', frame)
    WARN_POPUP('saved')
    os.chdir('../')


def main(window):
    b_getCap = 0
    filePath = 0
    b_isImg = False
    b_isVid = False
    b_GetContour = False
    bShow = False
    while True:
        event, values = window.read(timeout=50)
        scale = int(values['frameScale'])/100
        threshold = (int(values['thres1']), int(values['thres2']))

        if event == sg.WIN_CLOSED or event == 'Cancel' or event == 'Exit':
            break
        if event == 'Ok':
            b_getCap += 1
            if (values['filename'] != ''):
                filePath = str(values['filename'])
                filename, file_extension = os.path.splitext(filePath)
                inputType = values['inputType']
                Arr_img = ['.jpg', '.png']
                Arr_vid = ['.mp4', '.avi', '.mov']
                for x in Arr_img:
                    if file_extension == x:
                        b_isImg = True
                        b_isVid = False
                for x in Arr_vid:
                    if file_extension == x:
                        b_isImg = False
                        b_isVid = True
                bShow = True
            else:
                b_isVid = True
                bShow = True
        if bShow == True and b_isVid:
            print('filePath:'+str(filePath))
            # Vid Treatment
            if(b_getCap == 1):
                print(filePath)
                b_getCap = 0
                cap = cv2.VideoCapture(filePath)
                # cap = cv2.VideoCapture(filename)
            # Canny Threshold:
            print(str(values['inputType'])+'    ' +
                  str(int(values['thres1']))+'  '+str(int(values['thres2']))
                  + 'fileLocation:' + str(values['filename'])
                  )
            try:
                frame = utils.VID_1(cap, scale, (3, 3), int(
                    values['val_sigmaG']), threshold)
                imgbytes = cv2.imencode('.png', frame)[1].tobytes()
                window['FRAME'].update(data=imgbytes)
            except Exception as err:
                print("Uh oh, Warning: '" + str(err) + "'")
        if bShow == True and b_isImg:
            try:
                frame, unwrap = utils.IMG(scale, filePath, int(
                    values['val_sigmaG']), threshold, b_GetContour)
                print('getContour:  '+str(b_GetContour)+'*****************')
                if event == 'Show':
                    print('show-----------------------')
                    paper = utils.PAPER_EFFECT(unwrap)
                    SAVE_FILE(paper)
                if event == 'Save':
                    SAVE_FILE(unwrap)
                if event == 'ScanImg':
                    b_GetContour = not b_GetContour
            except Exception as err:
                print("Uh oh, Warning: '" + str(err) + "'")
            imgbytes = cv2.imencode('.png', frame)[1].tobytes()
            window['FRAME'].update(data=imgbytes)
    window.close()


search_section = [[
    sg.Column([
        [sg.Text('File name')],
        [sg.Input('', key='filename'), sg.FileBrowse()],
        [sg.Button('Ok'), sg.Button('Cancel')]],
        element_justification='c'
    ),
    sg.Column([
        [sg.Text('                      ')]],
        element_justification='c'
    ),
    sg.Column([
        [sg.Text('Combo Holder')],
        [sg.Combo(['1', '2'], default_value='1',
                  size=(20, 1), key='inputType')],
        [sg.Text('set Scale \'%\'')],
        [sg.Slider(range=(10, 100), default_value=50, size=(
            15, 10), orientation='horizontal', font=('Helvetica', 12), key='frameScale')]
    ],
        element_justification='c'
    )
]]
display_section = [
    [sg.Text('OpenCV Demo', size=(60, 1), justification='center')],
    [sg.Image(filename='', key='FRAME')],
]
action_section = [
    [sg.Text('                                      '), sg.Button('ScanImg')],
    [sg.Text('Show All Process Window'), sg.Button('Show')],
    [sg.Text('                                      '),
     sg.Button('Save')],
]
slider_section = [

    [sg.Text('Gauss_sigma'), sg.Slider(range=(0, 100),
                                       default_value=5,
                                       resolution=5,
                                       size=(30, 10),  # Width-Height
                                       orientation='horizontal',
                                       font=('Helvetica', 12),
                                       key='val_sigmaG')],
    [
        sg.Text('Threshold'),
        sg.Slider(range=(1, 200), default_value=50, resolution=5, size=(
            15, 10), orientation='horizontal', font=('Helvetica', 12), key='thres1'),
        sg.Slider(range=(1, 200), default_value=150, resolution=5, size=(
            15, 10), orientation='horizontal', font=('Helvetica', 12), key='thres2')
    ],
    [sg.Text('placeHdr'), sg.Slider(range=(1, 500),
                                    default_value=10,
                                    size=(30, 10),  # Width-Height
                                    orientation='horizontal',
                                    font=('Helvetica', 12),
                                    key='val_sharp')],
    [sg.Text('placeHdr'), sg.Slider(range=(11, 29),
                                    default_value=15,
                                    size=(30, 10),  # Width-Height
                                    tick_interval=5,
                                    orientation='horizontal',
                                    font=('Helvetica', 12),
                                    key='val_blur')],
]
slider_section = [[sg.Frame('SLIDERS', layout=slider_section)]]
action_section = [[sg.Frame('ACTIONS', layout=action_section)]]
util_section = slider_section+action_section
util_section2 = [[sg.Column(display_section, element_justification='l'),
                 sg.Column(util_section, element_justification='r')]]
layout = search_section + util_section2
window = sg.Window('Test', layout, location=(250, 0))
VidPath_default = 'vidE.mp4'
ImgPath_default = 'scanImg.png'

main(window)
