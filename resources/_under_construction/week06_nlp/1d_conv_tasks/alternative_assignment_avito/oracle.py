# -*- coding: utf-8 -*-
from __future__ import print_function
import numpy as np

def APatK ( y_true,y_predicted, K =32500):
    """Calculates AP@k given true Y and predictions (probabilities).
    Sorts answers by y_predicted to obtain ranking"""

    sort_by_ypred = np.argsort(-y_predicted)
    
    y_true = y_true[sort_by_ypred]
    y_predicted = y_predicted[sort_by_ypred]
    
    countRelevants = 0
    listOfPrecisions = []
    
    for i in range(min(K,len(y_true))):
        currentk = i + 1.0
        if y_true[i] !=0:
            countRelevants+=1
        precisionAtK = countRelevants / currentk 
        listOfPrecisions.append(precisionAtK)
    return np.sum( listOfPrecisions ) / min(K,len(y_true)) 





import sys
import socket

def score(final_accuracy,final_auc,final_apatk):
    
    print ("\nAUC:")
    if final_auc >= 0.99:
        print ("\tПиши статью. (great)")
    elif final_auc >= 0.97:
        print ("\tОтличное решение! (good)")
    elif final_auc >= 0.95:
        print ("\tСойдёт, хотя можно ещё поднажать (ok)")
    elif final_auc >= 0.9:
        print ("\tНеплохо, но ты можешь лучше! (not ok)")
    elif final_auc > 0.8:
        print ("\tТы на правильном пути! (not ok)")
    elif final_auc > 0.65:
        print ("\tДобавь жару! (not ok)")
    else:
        print ("\tМожет быть, она недоучилась? Ну или слишком маленькая? Или в детстве болела? (not ok)")
    
        
    print ("\nAccuracy:")
    if final_accuracy >= 0.97:
        print ("\tОчешуенно! (great)")
    elif final_accuracy >= 0.95:
        print ("\tОтличный результат! (good)")
    elif final_accuracy >= 0.9:
        print ("\tВсё ок (ok)")
    else:
        print ("Надо бы подтянуть. (not ok)")

    print ("\nAverage precision at K:")
    if final_apatk > 0.99:
        print ("\tЗасабмить на kaggle! (great) \n\t Нет, ну честно - выкачай avito_test.tsv, засабмить и скажи, что вышло.")
    elif final_apatk > 0.95:
        print ("\tОтличный результат (good)")
    elif final_apatk > 0.92:
        print ("\tВы побили baseline (ok)")
    else:
        print ("\tНадо бы поднажать (not ok)")
        
    if socket.gethostname().startswith("cv-gpu" ):
        if final_apatk>0.92 and final_accuracy >0.9 and final_auc > 0.95 :
            warn_them()
        else:
            print 

def warn_them():
    """warns some users of what is impending"""
    sys.stderr.write("""\nWe Are Watching You!
                                                      .             ..                           
                                    .      ...;c:,::..  '  ':oococ:. ..                          
                              ..    .':dodxkkxxxxxxxkxddxkkkkkkkkkxkkdl:.                        
                             .'':lodxxxdxxxxxddxxddodddxkkxdxxxxxxxxxxxxk:d;,.                   
                          .;dxkxxxxxxxxk000Okdooooooloodxxdddxxkkxxddddodkkkkd'  .               
                       .:lkkkkkkkkkkkOOOkkxxxxdollllllloooddxxkOOOOkxdddldkkkOxc.                
                     .lkkkOOOkkkkOOOkkxdollcccccccccccccccllddxkOOOOkddddddxxxddd;.              
                   .cxkkOOOOOOOOkkkdolc:::;;;;;;;;;;;;;::::ccclodxkOOxddooodddooodo'             
                  ,dxkkOOOOOOOOkxolc::;;;;;;;;;;;;;;;;;;;;::::cclodxkkxddooolooollooc..          
                 ,xxkkkOOO0OOkxdol:;;;;;;,,,,;;;;;;;;;;;;;;;;;::ccloxxxxdoooollllollod,          
                .xxkkkkOOOkxxollc;;;;,,,,,,,,,,,,,;,,;;;;;;;;;;::cclodxxddoollllcclllclc.        
                ;xkkxkOOkdoolllc;;;,,,,,,,,,,,,,,;;;,,;;;;;;;;;:::clloddddoooolccc::lcclo:       
               .ddkkxOOxdoolllc:;;;;;;;;;;;;,,,,,;;,,,;;;;;;;::::::ccodlcdddooolccc::cllll..     
              'xdddxxOOkxolclc::;;:coooooollllc:;;,,;;;;::ccoddddddoollc;coxkxoc:::cc:clcco.     
              oxddddxkxxdoc::::;:lolccccccccloolc;,,;:ccloddxxxkkOOOkdl::lxkkkxoc:;:cccccoo.     
              xdododdkkxdlc::c::coc::::clooollllc:;;:cllldxxxddddddxkkdodxkkkxxkdl:;::::odc      
             ,dddoodkkkdlc::cc::lccccldkkOOxdddol:;;:codkOO000OkdooodkkxxkOkxxkkkdlc:::;cd.      
             :odoodkkxolc:ccc::cllloxkdollloooddoc;;:lxO00OkxxxxkxdddkkkxkkkxkO0Okxolc;;:c .     
             oddodkOxolc:::c::::cloddo;;:c:;:llc::,,;lxOOko:;:cccdxxkOkkkkkxkkOO00kdllc;:o..     
             cddddkxdlc:::c::;;::cc:cc;:clccc::;;,,,;:oxkxc;:cloloxkkkkkxxkxdxollkxl;:cc:c;.     
             .oddxkkdc::cc::;;;;;;;;;cllllllcc:;,,,,;codxkdllooddxkxddddddxkooc::dOd:;:lolc'     
              'dddxxolcllcc:;;,,,,;;;;;;;:;;;;;;,,,,;:loddoccllodddlccclodxkxl:::dOdl::codlc     
               cxdddooodolo:;;,,,,,,,,,,,,,,,;;;,',,;:codol:;::::c::;:clodxxko::cdkolc;ccolc.    
                 cdxxxdkdodl:;,,,,,,,''''',::ll:;,;:llodddoc;;,;;;;::cclloxkkxlcokxolc:lcll:.    
                  ldoddkxodl::;,,,,,,,',;:c::ooccldxkO0K0kxxo:;,;;;:ccllclxkkxlcdkxol::dol:      
                  .dddoddoxc::;;,;;;,,;:cc;;;;;;:lxkkkkkkkkkko;;;;:cloooccxkxxocxkooocc;cd.      
                   ,cldddl:;;:;;;;;;;;:c:;,,,,,,,;;:cllodddxddc:::clooddlcxOxoclxkooool  :       
                     'xdd:;;;::;;;;;;;::;,,,,,,,,;;;;;:cccllllcc:cclooddl:xkxoloxxdodoc          
                  ..  cdxoccc::;;;;;;;;;;;;::cccc:ccllloooodolccc:clloodccxxdoodxxdodc .         
                       .;;  .::;;;;;;;;;:lxO0kdollodxxdkkOOOkdlcc;:cclodclodoodxdl:,.            
                            .c::;;;;;,;ccodOxlc:codkOOOOO0KK0xc:;;:cllddc,.do;.',.  .            
                           .clc::;;;;;;;;;::::::cclcllodxddooc,;;:clldxo:..;,                    
                         .;clllcc:::;;,,,,;;;;;;;::cccclodddl;;;:looxkxl.                        
                        .c:ccooclcc::;;,,,;;;:::ccllooddddol:::coxxkkkc .  .                     
                        :l:clodccllcc:;;;;,;;;;:::cccccc:::cclodkkkkx:                           
                       ;ll:clddcccloolc:;;,,,,;,,,,;;:::::cldxkkkOkc                             
                      .ddcclld,lcclloooolc;;;,,,;;;;:ccclldxxdc;,;.                              
                      :docclod,'lccclloddxdollclllloooool:;.                                     
                     .dxlcclldo :cccllloodxkd.   ....                                            
                     ;dxlcclodx.'lcclllloodxx.                     
 \n""")
    sys.stderr.write("""
                           ______________________________________
                 _\|/^    /       Молодцы, а теперь слезайте     \    \|| /
                  (_oo   /              с казённой GPU            \   oo   /  
                   |     \________________________________________/  О_    --
                  /|\                                                 )    =
                   |                                                 (.   --
                   LL                                                 1 1\ 
                mborisyak@                                           jheuristic@
    """)

