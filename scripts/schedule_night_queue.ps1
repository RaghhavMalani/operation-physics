# Register the Windows scheduled task that starts the night queue in WSL (DECISIONS.md D128). Unlike a waiter inside WSL,
# it survives a WSL VM stop: wsl.exe boots the VM if it is down. queue_r4_seeds.sh holds a flock, so a second start (the
# waiter, or this task) logs "already running" and exits 0.
#
# wsl.exe runs the queue in the foreground, so the task shows "Running" all night. A wsl.exe that exits at once after
# `nohup ... &` leaves the instance with no client, and WSL may stop it when idle, killing the queue (D128).
#
# From WSL:  powershell.exe -NoProfile -ExecutionPolicy Bypass -File "$(wslpath -w scripts/schedule_night_queue.ps1)"
# Show:      Get-ScheduledTask -TaskName physcal-night-queue | Format-List *   (or: schtasks /query /tn physcal-night-queue /v /fo LIST)
# Delete:    Unregister-ScheduledTask -TaskName physcal-night-queue -Confirm:$false   (or: schtasks /delete /tn physcal-night-queue /f)
param(
  [string]$At = '23:00',                               # local time (India Standard Time), today
  [string]$TaskName = 'physcal-night-queue',
  [string]$Distro = 'Ubuntu',
  [string]$QueueEnv = 'GPU_DEADLINE=09:00 MC_B=1',
  [string]$Log = 'results/queue_r4_seeds/nohup.log'
)
$ErrorActionPreference = 'Stop'
$bash = "cd ~/operation-physics && $QueueEnv nohup scripts/queue_r4_seeds.sh >> $Log 2>&1"
$action = New-ScheduledTaskAction -Execute 'wsl.exe' -Argument "-d $Distro -e bash -lc `"$bash`""
$trigger = New-ScheduledTaskTrigger -Once -At (Get-Date $At)
$principal = New-ScheduledTaskPrincipal -UserId "$env:USERDOMAIN\$env:USERNAME" -LogonType Interactive  # only when logged on
$settings = New-ScheduledTaskSettingsSet -DontStopIfGoingOnBatteries -ExecutionTimeLimit (New-TimeSpan -Hours 16)
Register-ScheduledTask -TaskName $TaskName -Action $action -Trigger $trigger -Principal $principal -Settings $settings `
  -Description 'physcal: night GPU queue (scripts/queue_r4_seeds.sh), one-time; DECISIONS.md D128' | Out-Null
$t = Get-ScheduledTask -TaskName $TaskName
"task:      $($t.TaskName) ($($t.State))"
"runs:      $($t.Actions[0].Execute) $($t.Actions[0].Arguments)"
"trigger:   once at $($t.Triggers[0].StartBoundary)"
"principal: $($t.Principal.UserId), logon type $($t.Principal.LogonType)"
"settings:  start on batteries $(-not $t.Settings.DisallowStartIfOnBatteries), stop on batteries $($t.Settings.StopIfGoingOnBatteries), time limit $($t.Settings.ExecutionTimeLimit)"
"next run:  $((Get-ScheduledTaskInfo -TaskName $TaskName).NextRunTime)"
