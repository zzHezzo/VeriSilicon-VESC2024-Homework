/*
 * zzh_demo.c
 *
 *  Created on: 2024年5月12日
 *      Author: 28065
 */
#include "uart_printf.h"
#include "osal_task_api.h"
#include "board.h"
#include "vpi_event.h"
#include <inttypes.h>
#include "vs_conf.h"
#include "zzh_demo.h"

typedef struct {
    uint32_t value;          // 原有的整数值
    const char* message; // 新增的字符串消息
} EventData;

/*FreeRTOS配置*/
 /*
  * task_init_app 任务 配置
  * 包括: 任务句柄 任务优先级 堆栈大小 创建任务
  */
#define INIT_TASK_PRIO         1            //任务优先级
#define INIT_STK_SIZE          512          //任务堆栈大小
TaskHandle_t InitTask_Handler;              //任务句柄
static void task_init_app(void *pvParameters);         //任务函数

/*
 * task_sys_mgr 任务 配置
 * 包括: 任务句柄 任务优先级 堆栈大小 创建任务
 */
#define SYS_TASK_PRIO         4            //任务优先级
#define SYS_STK_SIZE          256          //任务堆栈大小
TaskHandle_t SysTask_Handler;              //任务句柄
static void task_sys_mgr(void *pvParameters);         //任务函数

/*
 * task_custom_mgr 任务 配置
 * 包括: 任务句柄 任务优先级 堆栈大小 创建任务
 */
#define CUSTOM_TASK_PRIO         5            //任务优先级
#define CUSTOM_STK_SIZE          256          //任务堆栈大小
TaskHandle_t CustomTask_Handler;              //任务句柄
static void task_custom_mgr(void *pvParameters);         //任务函数


void homework1_printf_hellovs(void)
{
    uart_printf("Hello VeriSilicon\r\n");
}

void system_create_InitTask(void)
{
    //创建初始化任务
	InitTask_Handler = osal_create_task(task_init_app,
			                            "init_app",
										INIT_STK_SIZE,
										INIT_TASK_PRIO,
										NULL);
	osal_start_scheduler();
}

static void task_init_app(void *param)
{
	BoardDevice board_dev;
	/*Initialize board*/
    board_register(board_get_ops());
    board_init((void *)&board_dev);
	if (board_dev.name)
	    uart_printf("Board: %s", board_dev.name);

    uart_printf("系统任务初始化ing\r\n");
    taskENTER_CRITICAL();           /* 进入临界区 */

    SysTask_Handler = osal_create_task(task_sys_mgr,
			                            "sys_name",
										SYS_STK_SIZE,
										SYS_TASK_PRIO,
										NULL);

    CustomTask_Handler = osal_create_task(task_custom_mgr,
			                            "custom_name",
										CUSTOM_STK_SIZE,
										CUSTOM_TASK_PRIO,
										NULL);

    osal_delete_task(NULL);
    taskEXIT_CRITICAL();            /* 退出临界区 */
}

/**
 * @brief Event handler definition
 * @param cobj The pointer of manager
 * @param event_id The event need to be handled
 * @param param The pointer of param for event
 */
static int sys_event_handler(void *cobj,uint32_t event_id,void *param)
{
	EventData* event_data = (EventData*)param;
	switch(event_id){
	case EVENT_SYS_TEST:
		uart_printf("Received EVENT_SYS_TEST with value: %ld and message: %s\n",
				    event_data->value,event_data->message);
		break;
	default:
		uart_printf("Unhandled event type\n");
		break;
	}
	return 0; // 返回 0 表示成功处理事件
}

static void task_sys_mgr(void *pvParameters)
{
    static void * pSysManager;
    pSysManager = vpi_event_new_manager(COBJ_SYS_MGR,sys_event_handler);
    if(vpi_event_register(EVENT_SYS_TEST,pSysManager) == EVENT_OK)
    {
    	uart_printf("event register successful!\r\n");
    	while(1){
    			if(vpi_event_listen(pSysManager) != EVENT_OK)
    				{
    				    uart_printf("Error listening for events\n");
    					break;  // 或处理错误
    				}
    		}
    }else{
    	uart_printf("event register error!\r\n");
    }
	osal_delete_task(NULL);
}

static void task_custom_mgr(void *pvParameters)
{
	static EventData event_data = {0xa5a5,"Hello from sender!"};
	while(1){
		osal_sleep(1000);
    	vpi_event_notify(EVENT_SYS_TEST,&event_data);
    	uart_printf("event notify!!!\n");
    }
	osal_delete_task(NULL);
}


