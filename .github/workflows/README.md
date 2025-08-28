# GitHub Actions Workflows

## Daily Trade Execution Workflow

This workflow runs your trading script (`trade.sh`) at 3:55 PM New York time every weekday and emails you the results.

### Setup Instructions

1. **Configure Email Secrets in GitHub Repository**
   - Go to your repository on GitHub
   - Navigate to Settings → Secrets and variables → Actions
   - Click **New repository secret**
   - Add the following secrets:

     | Secret Name | Description | Example Value |
     |-------------|-------------|---------------|
     | `EMAIL_USERNAME` | Your Gmail address | `your.email@gmail.com` |
     | `EMAIL_PASSWORD` | Gmail App Password (see below) | `your-app-password` |
     | `NOTIFICATION_EMAIL` | Email address to receive notifications | `your.email@gmail.com` |

2. **Generate Gmail App Password**
   - Enable 2-Factor Authentication on your Gmail account
   - Go to Google Account settings → Security → App passwords
   - Generate an app password for "Mail"
   - Use this app password (not your regular Gmail password) as the `EMAIL_PASSWORD` secret

### Workflow Details

- **Schedule**: Scheduled to run around 3:55 PM Eastern Time (ET) every weekday (Monday-Friday), with validation for the entire 3 PM hour
- **Execution**: Runs the trading script directly with proper path handling
- **Notification**: Sends email with trade results only when execution actually occurs
- **Timezone Handling**: Uses precise America/New_York timezone checking to validate we're within the 3 PM ET hour, automatically handling DST transitions

### Manual Execution

You can also trigger the workflow manually:
1. Go to the Actions tab in your repository
2. Select "Daily Trade Execution" workflow
3. Click "Run workflow"
