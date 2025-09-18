import boto3
import time

sagemaker = boto3.client('sagemaker')

def stop_all_training_jobs():
    """
    Finds and stops all SageMaker training jobs that are 'InProgress'.
    """
    print("\n⏸️  Stopping all in-progress training jobs...")
    
    paginator = sagemaker.get_paginator('list_training_jobs')
    jobs_found = 0
    
    for page in paginator.paginate(StatusEquals='InProgress'):
        for job in page.get('TrainingJobSummaries', []):
            jobs_found += 1
            job_name = job['TrainingJobName']
            job_status = job['TrainingJobStatus']
            print(f"Found running job: {job_name} (status: {job_status})")

            try:
                sagemaker.stop_training_job(TrainingJobName=job_name)
                print(f"  - Stopping job: {job_name}")
            except Exception as e:
                print(f"  ❌ ERROR stopping {job_name}: {str(e)}")

    if jobs_found == 0:
        print("No in-progress training jobs found.")


def delete_sagemaker_experiments():
    """
    Deletes all SageMaker Experiments, Trials, and Trial Components from the top down.
    """
    print("\n🧹 Starting SageMaker Experiment cleanup...")

    # ✅ ADDED: Delete the top-level Experiment first. This is the most robust method.
    try:
        paginator = sagemaker.get_paginator('list_experiments')
        for page in paginator.paginate():
            for exp_summary in page.get('ExperimentSummaries', []):
                exp_name = exp_summary['ExperimentName']
                print(f"  - Deleting Experiment: {exp_name}")
                try:
                    # To delete an experiment, you must first disassociate its children.
                    # This logic remains as a robust way to clean up before deleting the parent.
                    trial_paginator = sagemaker.get_paginator('list_trials')
                    for trial_page in trial_paginator.paginate(ExperimentName=exp_name):
                        for trial_summary in trial_page.get('TrialSummaries', []):
                            trial_name = trial_summary['TrialName']
                            for comp_summary in sagemaker.list_trial_components(TrialName=trial_name)['TrialComponentSummaries']:
                                sagemaker.disassociate_trial_component(TrialName=trial_name, TrialComponentName=comp_summary['TrialComponentName'])
                    
                    sagemaker.delete_experiment(ExperimentName=exp_name)
                except Exception as e:
                    print(f"    ❌ ERROR deleting experiment {exp_name}: {str(e)}")
    except Exception as e:
        print(f"Could not list or delete experiments: {e}")

    # Fallback cleanup for any orphaned trials and components
    print("\n⏳ Waiting 30 seconds for AWS to process experiment deletions...")
    time.sleep(30)
    
    # --- Fallback: Delete any orphaned Trials ---
    try:
        paginator = sagemaker.get_paginator('list_trials')
        for page in paginator.paginate():
            for trial_summary in page.get('TrialSummaries', []):
                trial_name = trial_summary['TrialName']
                print(f"  - Deleting orphaned Trial: {trial_name}")
                try:
                    sagemaker.delete_trial(TrialName=trial_name)
                except Exception as e:
                    print(f"    ❌ ERROR deleting trial {trial_name}: {str(e)}")
    except Exception as e:
        print(f"Could not list or delete trials: {e}")

    # --- Fallback: Delete any orphaned Trial Components ---
    try:
        paginator = sagemaker.get_paginator('list_trial_components')
        for page in paginator.paginate():
            for summary in page.get('TrialComponentSummaries', []):
                name = summary['TrialComponentName']
                print(f"  - Deleting orphaned Trial Component: {name}")
                try:
                    sagemaker.delete_trial_component(TrialComponentName=name)
                except Exception as e:
                    print(f"    ❌ ERROR deleting component {name}: {str(e)}")
    except Exception as e:
         print(f"Could not list or delete trial components: {e}")


def delete_endpoints():
    """
    Finds and deletes all SageMaker Endpoints and their associated configs.
    """
    print("\n🗑️ Starting endpoint cleanup...")

    paginator = sagemaker.get_paginator('list_endpoints')
    endpoints_found = 0
    
    for page in paginator.paginate():
        for ep in page.get('Endpoints', []):
            endpoints_found += 1
            ep_name = ep['EndpointName']
            ep_status = ep['EndpointStatus']
            print(f"Found endpoint: {ep_name} (status: {ep_status})")

            if ep_status != 'Deleting':
                try:
                    ep_config_name = sagemaker.describe_endpoint(EndpointName=ep_name)['EndpointConfigName']
                    sagemaker.delete_endpoint(EndpointName=ep_name)
                    print(f"  - Deleting endpoint: {ep_name}")
                    sagemaker.delete_endpoint_config(EndpointConfigName=ep_config_name)
                    print(f"  - Deleting endpoint config: {ep_config_name}")
                except Exception as e:
                    print(f"  ❌ ERROR deleting {ep_name}: {str(e)}")
    
    if endpoints_found == 0:
        print("No active endpoints found.")


if __name__ == "__main__":
    stop_all_training_jobs()
    delete_sagemaker_experiments()
    delete_endpoints()
    print("\n✅ SageMaker cleanup completed.")