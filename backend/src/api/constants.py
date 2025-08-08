AI_ANALYSIS_PROMPT_FORMAT = """
Analyze the current market conditions for {asset_name} based on the following technical indicators:

**Current Market Data:**
- Price: ${current_price:.2f} ({price_change_pct:+.2f}% today)
- RSI: {rsi:.1f} ({rsi_interpretation})
- MACD: {macd:.4f}
- Bollinger Band Position: {bollinger_position:.1f}%
- Average True Range: ${average_true_range:.2f}
- Combined Signal: {signal_interpretation}
- Support Levels: ${support_level_0:.2f}, ${support_level_1:.2f}
- Resistance Levels: ${resistance_level_0:.2f}, ${resistance_level_1:.2f}

**Instructions:**
Provide a concise market analysis in 2-3 paragraphs that focuses on:
1. Current market sentiment and momentum based on the technical indicators
2. Key price levels to watch (support/resistance) and potential trading opportunities
3. Risk assessment and what traders should be cautious about

**Format Requirements:**
- Write in professional, accessible language suitable for traders
- Use markdown formatting for emphasis. Specifically, use *italic* for key terms and **bold** for important levels
- Do not include headers or titles
- Keep the analysis practical and actionable
- Focus on insights that go beyond just restating the numbers
"""


