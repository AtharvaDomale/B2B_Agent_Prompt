import json
import os
import sys
from typing import Any, Dict, List, Optional

import streamlit as st
from agno.agent import Agent
from agno.memory.v2 import Memory
from agno.models.openai import OpenAIChat
from agno.tools.exa import ExaTools


def require_env(var_name: str) -> None:
    if not os.getenv(var_name):
        print(f"Error: {var_name} not set. export {var_name}=...")
        sys.exit(1)


def create_company_finder_agent() -> Agent:
    exa_tools = ExaTools(category="company")
    memory = Memory()
    return Agent(
        model=OpenAIChat(id="gpt-5"),
        tools=[exa_tools],
        memory=memory,
        add_history_to_messages=True,
        num_history_responses=6,
        session_id="gtm_outreach_company_finder",
        show_tool_calls=True,
        instructions=[
            "You are CompanyFinderAgent, an expert at identifying high-value B2B prospects using advanced search techniques.",
            "",
            "SEARCH STRATEGY:",
            "- Use multiple search queries with different angles (industry keywords, technology stack, company size indicators, recent news/funding)",
            "- Search for companies mentioned in industry reports, tech blogs, and funding announcements",
            "- Look for companies posting relevant job openings (indicates growth/investment in the area)",
            "- Search for companies mentioned alongside competitors or in comparison articles",
            "",
            "QUALIFICATION CRITERIA:",
            "- Company size: 50-5000 employees (unless specified otherwise)",
            "- Active online presence and professional website",
            "- Clear business model and revenue generation",
            "- Recent activity (news, product launches, hiring, funding within 18 months)",
            "- Strong alignment with targeting criteria (reject poor fits)",
            "",
            "QUALITY REQUIREMENTS:",
            "- Each company must have a clear, compelling fit reason",
            "- Prioritize companies showing growth indicators (hiring, funding, expansion)",
            "- Avoid companies that are too small (<10 employees) or too enterprise (>10k employees) unless specifically requested",
            "- Verify company is still active and operational",
            "",
            "OUTPUT FORMAT:",
            "Return ONLY valid JSON with key 'companies' as a list. Respect the exact limit requested.",
            "Each item must include:",
            "- name: Company legal name",
            "- website: Full URL (https://...)",
            "- why_fit: 2-3 specific, compelling sentences explaining the fit",
            "- employee_count: Estimated number (e.g., '100-500')",
            "- growth_signals: List of recent indicators (funding, hiring, product launches, etc.)",
            "",
            "Be selective and thorough. Quality over quantity."
        ],
    )


def create_contact_finder_agent() -> Agent:
    exa_tools = ExaTools()
    memory = Memory()
    return Agent(
        model=OpenAIChat(id="gpt-4o"),
        tools=[exa_tools],
        memory=memory,
        add_history_to_messages=True,
        num_history_responses=6,
        session_id="gtm_outreach_contact_finder",
        show_tool_calls=True,
        instructions=[
            "You are ContactFinderAgent, a specialist in identifying and locating key decision makers for B2B outreach.",
            "",
            "PRIMARY TARGET ROLES (in priority order):",
            "1. GTM Leadership: VP/Director Marketing, Growth, Demand Gen, Revenue Operations",
            "2. Sales Leadership: VP/Director Sales, Sales Development, Business Development",
            "3. Strategic Roles: Chief of Staff, Director Strategy, Head of Partnerships",
            "4. Talent/People Ops: Director/VP Talent Acquisition, Head of People, Chief People Officer",
            "5. Product Marketing: Director/VP Product Marketing, Growth Product Manager",
            "6. Executive level: CEO, COO, CMO, CRO (only if relevant to offering)",
            "",
            "SEARCH METHODOLOGY:",
            "- Start with LinkedIn search: 'site:linkedin.com \"[Company Name]\" \"[Target Title]\"'",
            "- Check company About/Team pages for leadership bios",
            "- Search for recent press releases, podcast appearances, speaking engagements",
            "- Look for company blog posts, case studies, or whitepapers authored by employees",
            "- Check industry event speaker lists and conference attendees",
            "- Search for company employees quoted in news articles or industry publications",
            "",
            "EMAIL DISCOVERY STRATEGY:",
            "1. Direct sources: Company contact pages, team pages, press releases, author bios",
            "2. Pattern inference: Identify company email format from known emails, then apply pattern",
            "3. Common patterns: first.last@domain, f.last@domain, firstl@domain, first@domain",
            "4. Verification: Cross-check inferred emails against multiple sources when possible",
            "",
            "CONTACT QUALITY REQUIREMENTS:",
            "- Focus on mid-to-senior level roles (Director+ or equivalent influence)",
            "- Prioritize recently active contacts (LinkedIn activity, recent mentions, current employment)",
            "- Each contact must have clear decision-making authority or strong influence in relevant area",
            "- Aim for 2-4 contacts per company, but prefer quality over quantity",
            "",
            "EMAIL VERIFICATION:",
            "- Mark emails as inferred=true when using pattern matching",
            "- Only include professional email domains (company domain, not gmail/yahoo/etc)",
            "- Double-check email format consistency within same company",
            "",
            "OUTPUT FORMAT:",
            "Return ONLY valid JSON: {\"companies\": [{\"name\": \"Company Name\", \"contacts\": [{\"full_name\": \"John Smith\", \"title\": \"VP Marketing\", \"email\": \"john.smith@company.com\", \"inferred\": false, \"source\": \"company website\", \"last_activity\": \"LinkedIn post 2 weeks ago\"}]}]}",
            "",
            "CRITICAL: Avoid generic CEO/founder contacts unless no other relevant roles exist. Focus on practitioners who would use or evaluate the offering."
        ],
    )


def create_phone_finder_agent() -> Agent:
    exa_tools = ExaTools()
    memory = Memory()
    return Agent(
        model=OpenAIChat(id="gpt-4o"),
        tools=[exa_tools],
        memory=memory,
        add_history_to_messages=True,
        num_history_responses=6,
        session_id="gtm_outreach_phone_finder",
        show_tool_calls=True,
        instructions=[
            "You are PhoneFinderAgent, an expert at locating professional phone numbers through comprehensive web research.",
            "",
            "SEARCH METHODOLOGY:",
            "1. Company sources: Official websites, contact pages, press releases, team pages",
            "2. Professional directories: LinkedIn contact info, ZoomInfo, industry directories",
            "3. News/Media: Press release contact info, media quotes with phone numbers",
            "4. Conference/Event listings: Speaker contact information",
            "5. Business registration records: SEC filings, corporate directories",
            "6. Social proof: Professional bios mentioning phone numbers",
            "",
            "SEARCH QUERIES TO USE:",
            "- '[Company Name] [Contact Name] phone number'",
            "- '[Company Name] contact directory phone'",
            "- 'site:linkedin.com \"[Contact Name]\" phone'",
            "- '[Contact Name] [Title] [Company] contact phone'",
            "- '[Company Name] press contact phone number'",
            "- '[Company Name] media kit contact information'",
            "",
            "PHONE NUMBER VALIDATION:",
            "- Verify format matches country/region standards",
            "- Direct numbers: 10+ digits, proper area code",
            "- Office numbers: Often have extensions",
            "- Mobile numbers: Typically follow standard mobile formats",
            "- Include country code when available (+1 for US, etc.)",
            "",
            "PRIORITIZATION (most to least valuable):",
            "1. Direct dial numbers (desk phones)",
            "2. Mobile/cell numbers", 
            "3. Office main numbers with extensions",
            "4. General office numbers",
            "",
            "VERIFICATION LEVELS:",
            "- verified=true: Found on official company sources, press releases, or professional directories",
            "- verified=false: Found on unofficial sources, social media, or inferred",
            "",
            "OUTPUT FORMAT:",
            "Return ONLY valid JSON: {\"companies\": [{\"name\": \"Company Name\", \"contacts\": [{\"full_name\": \"John Smith\", \"phone_number\": \"+1-555-123-4567\", \"phone_type\": \"direct\", \"verified\": true, \"source\": \"company contact page\"}]}]}",
            "",
            "NOTE: Phone finding has lower success rates than email - focus on quality and legitimate sources. Skip if no credible numbers found."
        ],
    )


def get_email_style_instruction(style_key: str) -> str:
    styles = {
        "Professional": [
            "STYLE: Professional B2B communication",
            "- Formal but personable tone",
            "- Clear structure: opening hook → value proposition → social proof → CTA",
            "- Use industry-appropriate terminology",
            "- Respectful and consultative approach",
            "- Length: 120-180 words"
        ],
        "Casual": [
            "STYLE: Casual but professional",
            "- Conversational, friendly tone",
            "- First-name basis, approachable language",
            "- Shorter paragraphs, easier to scan",
            "- Still maintain professional credibility",
            "- Length: 100-150 words"
        ],
        "Cold": [
            "STYLE: High-impact cold email",
            "- Strong pattern interrupt in first 15 words",
            "- Specific, quantifiable value proposition",
            "- Social proof or credibility indicator",
            "- Clear, specific CTA with low friction",
            "- Length: 80-120 words (shorter is better)"
        ],
        "Consultative": [
            "STYLE: Insight-led consultative approach",
            "- Lead with industry insight or trend observation",
            "- Frame potential challenges/opportunities",
            "- Position as thought partnership, not sales",
            "- Soft CTA focused on discussion/exchange",
            "- Length: 130-180 words"
        ]
    }
    return "\n".join(styles.get(style_key, styles["Professional"]))


def create_email_writer_agent(style_key: str = "Professional") -> Agent:
    memory = Memory()
    style_instruction = get_email_style_instruction(style_key)
    return Agent(
        model=OpenAIChat(id="gpt-5"),
        tools=[],
        memory=memory,
        add_history_to_messages=True,
        num_history_responses=6,
        session_id="gtm_outreach_email_writer",
        show_tool_calls=False,
        instructions=[
            "You are EmailWriterAgent, an expert B2B email copywriter specializing in personalized outreach that drives responses.",
            "",
            style_instruction,
            "",
            "PERSONALIZATION REQUIREMENTS:",
            "- Use 1-2 specific insights from research (company news, initiatives, challenges)",
            "- Reference specific company details (not generic industry observations)", 
            "- Connect insights directly to value proposition",
            "- Avoid obvious research like 'I see you're in the X industry'",
            "",
            "EMAIL STRUCTURE:",
            "1. Subject line: Specific, intriguing, avoid spam triggers",
            "2. Opening: Personal connection or specific insight",
            "3. Value proposition: Clear benefit relevant to their role",
            "4. Social proof: Brief credibility indicator (client, result, etc.)",
            "5. CTA: Specific ask with calendar link if provided",
            "",
            "SUBJECT LINE BEST PRACTICES:",
            "- 25-50 characters ideal",
            "- Include company name or specific reference",
            "- Question format often works well",
            "- Avoid: Free, Urgent, $ symbols, ALL CAPS",
            "- Examples: 'Quick question about [Company]', '[Insight] at [Company]', 'Help with [specific challenge]?'",
            "",
            "WRITING QUALITY:",
            "- Use active voice and specific language",
            "- Avoid corporate jargon and buzzwords",
            "- One main idea per paragraph",
            "- Include specific numbers/metrics when possible",
            "- End with clear next step",
            "",
            "PERSONALIZATION SOURCES:",
            "- Recent company news, funding, product launches",
            "- Leadership changes or new hires",
            "- Industry challenges or opportunities",
            "- Technology stack or tools mentioned",
            "- Growth indicators or expansion plans",
            "",
            "OUTPUT FORMAT:",
            "Return ONLY valid JSON: {\"emails\": [{\"company\": \"Company Name\", \"contact\": \"Contact Name\", \"subject\": \"Subject Line\", \"body\": \"Email body with \\n for line breaks\", \"personalization_used\": \"Brief note on what insight was used\"}]}",
            "",
            "CRITICAL: Each email must feel genuinely researched and personally written. Avoid template-like language."
        ],
    )


def create_research_agent() -> Agent:
    """Agent to gather interesting insights from company websites and Reddit."""
    exa_tools = ExaTools()
    memory = Memory()
    return Agent(
        model=OpenAIChat(id="gpt-5"),
        tools=[exa_tools],
        memory=memory,
        add_history_to_messages=True,
        num_history_responses=6,
        session_id="gtm_outreach_researcher",
        show_tool_calls=True,
        instructions=[
            "You are ResearchAgent, a specialist in gathering compelling, actionable insights for B2B outreach personalization.",
            "",
            "RESEARCH SOURCES (in priority order):",
            "1. Company website: About page, blog posts, case studies, product updates, leadership bios",
            "2. Recent news: Press releases, funding announcements, leadership changes, product launches",
            "3. Social proof: Customer testimonials, case studies, partnership announcements",
            "4. Reddit discussions: 'site:reddit.com [company name]' for authentic opinions and discussions",
            "5. Industry publications: Recent mentions, awards, rankings",
            "",
            "SEARCH STRATEGIES:",
            "- '[Company name] news 2024' OR '[Company name] news 2025'",
            "- '[Company name] funding series growth expansion'",
            "- '[Company name] product launch new feature'",
            "- 'site:reddit.com \"[company name]\"' (for authentic user discussions)",
            "- '[Company name] partnership collaboration'",
            "- '[Company name] hiring jobs remote'",
            "",
            "INSIGHT QUALITY CRITERIA:",
            "- Recent (within 6-12 months preferred)",
            "- Specific and factual (not generic industry trends)",
            "- Relevant to potential business impact",
            "- Shows company momentum, challenges, or opportunities",
            "- Can be naturally referenced in outreach",
            "",
            "INSIGHT CATEGORIES TO PRIORITIZE:",
            "1. Growth signals: Funding, hiring, expansion, new offices",
            "2. Product developments: New features, integrations, platform changes",
            "3. Market positioning: New verticals, customer segments, use cases",
            "4. Leadership: New hires, promotions, strategic appointments",
            "5. Challenges/Pain points: From Reddit, reviews, or news coverage",
            "6. Recognition: Awards, rankings, industry mentions",
            "",
            "REDDIT-SPECIFIC RESEARCH:",
            "- Look for authentic user experiences with the company/product",
            "- Identify common praise points or criticism",
            "- Find discussions about company culture, hiring, or challenges",
            "- Note any viral mentions or community discussions",
            "",
            "OUTPUT REQUIREMENTS:",
            "- 3-5 insights per company maximum",
            "- Each insight should be 1-2 sentences",
            "- Include source type (website, news, reddit, etc.)",
            "- Focus on insights that show genuine research effort",
            "- Avoid generic insights that could apply to any company",
            "",
            "OUTPUT FORMAT:",
            "Return ONLY valid JSON: {\"companies\": [{\"name\": \"Company Name\", \"insights\": [\"Recently raised $X Series B to expand into European markets (TechCrunch)\", \"Reddit users praise their customer support response time improvements\", \"Just launched AI-powered features after 6-month development cycle\"]}]}",
            "",
            "CRITICAL: Insights must be specific enough that mentioning them proves genuine research, not generic industry knowledge."
        ],
    )


def extract_json_or_raise(text: str) -> Dict[str, Any]:
    """Extract JSON from a model response with improved error handling."""
    try:
        # First try direct parsing
        return json.loads(text)
    except json.JSONDecodeError:
        # Look for JSON blocks in markdown
        import re
        json_pattern = r'```(?:json)?\s*(\{.*?\})\s*```'
        match = re.search(json_pattern, text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                pass
        
        # Look for any JSON-like structure
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            candidate = text[start:end + 1]
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                pass
        
        raise ValueError(f"Failed to parse JSON from response. Response was:\n{text[:500]}...")


def run_company_finder(agent: Agent, target_desc: str, offering_desc: str, max_companies: int) -> List[Dict[str, Any]]:
    prompt = (
        f"MISSION: Find exactly {max_companies} high-quality B2B prospect companies that are strong fits for our offering.\n\n"
        f"TARGET CRITERIA:\n{target_desc}\n\n"
        f"OUR OFFERING:\n{offering_desc}\n\n"
        f"REQUIREMENTS:\n"
        f"- Find companies actively growing or investing in relevant areas\n"
        f"- Each company must have 50+ employees (unless targeting SMB specifically)\n"
        f"- Strong website presence and professional operations\n"
        f"- Clear business model and revenue generation\n"
        f"- Active within last 18 months (news, hiring, product updates)\n\n"
        f"For each company, provide: name, website, why_fit (compelling 2-3 sentence explanation), employee_count, growth_signals.\n\n"
        f"Focus on quality over quantity. Reject poor fits."
    )
    resp = agent.run(prompt)
    data = extract_json_or_raise(str(resp.content))
    companies = data.get("companies", [])
    return companies[:max_companies]


def run_contact_finder(agent: Agent, companies: List[Dict[str, Any]], target_desc: str, offering_desc: str) -> List[Dict[str, Any]]:
    prompt = (
        f"MISSION: Find 2-4 high-quality decision makers per company who would evaluate, influence, or champion our offering.\n\n"
        f"TARGET CONTEXT:\n{target_desc}\n\n"
        f"OUR OFFERING:\n{offering_desc}\n\n"
        f"COMPANIES TO RESEARCH:\n{json.dumps(companies, indent=2)}\n\n"
        f"CONTACT REQUIREMENTS:\n"
        f"- Director level or above (or equivalent influence)\n"
        f"- Active on LinkedIn or mentioned in recent company content\n"
        f"- Clear connection to our offering area\n"
        f"- Professional email discoverable or inferable\n\n"
        f"For each contact found, verify current employment and activity level.\n"
        f"Return format: {{\"companies\": [{{\"name\": \"Company\", \"contacts\": [{{\"full_name\": \"Name\", \"title\": \"Title\", \"email\": \"email@company.com\", \"inferred\": false, \"source\": \"source\", \"last_activity\": \"description\"}}]}}]}}"
    )
    resp = agent.run(prompt)
    data = extract_json_or_raise(str(resp.content))
    return data.get("companies", [])


def run_phone_finder(agent: Agent, contacts_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    prompt = (
        f"MISSION: Find professional phone numbers for the contacts below using comprehensive web research.\n\n"
        f"CONTACTS TO RESEARCH:\n{json.dumps(contacts_data, indent=2)}\n\n"
        f"SEARCH PRIORITIES:\n"
        f"1. Direct dial numbers from company websites/directories\n"
        f"2. Mobile numbers from professional profiles\n"
        f"3. Office numbers with extensions\n"
        f"4. General office numbers\n\n"
        f"VERIFICATION:\n"
        f"- Mark verified=true only for official company sources\n"
        f"- Include country codes when available\n"
        f"- Validate number format and length\n\n"
        f"Return format: {{\"companies\": [{{\"name\": \"Company\", \"contacts\": [{{\"full_name\": \"Name\", \"phone_number\": \"+1-555-123-4567\", \"phone_type\": \"direct\", \"verified\": true, \"source\": \"source\"}}]}}]}}"
    )
    resp = agent.run(prompt)
    data = extract_json_or_raise(str(resp.content))
    return data.get("companies", [])


def run_research(agent: Agent, companies: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    prompt = (
        f"MISSION: Gather 3-5 specific, recent insights per company that would demonstrate genuine research in outreach emails.\n\n"
        f"COMPANIES TO RESEARCH:\n{json.dumps(companies, indent=2)}\n\n"
        f"RESEARCH OBJECTIVES:\n"
        f"- Find recent news, developments, or changes (last 12 months)\n"
        f"- Identify growth signals, challenges, or opportunities\n"
        f"- Discover authentic opinions from Reddit or forums\n"
        f"- Look for specific details that prove genuine research\n\n"
        f"INSIGHT QUALITY:\n"
        f"- Specific to this company (not generic industry trends)\n"
        f"- Recent and relevant to business decisions\n"
        f"- Could naturally be referenced in personalized outreach\n"
        f"- Shows company momentum or strategic direction\n\n"
        f"Return format: {{\"companies\": [{{\"name\": \"Company\", \"insights\": [\"Specific insight with context and source\"]}}]}}"
    )
    resp = agent.run(prompt)
    data = extract_json_or_raise(str(resp.content))
    return data.get("companies", [])


def run_email_writer(agent: Agent, contacts_data: List[Dict[str, Any]], research_data: List[Dict[str, Any]], 
                    offering_desc: str, sender_name: str, sender_company: str, calendar_link: Optional[str]) -> List[Dict[str, str]]:
    
    # Merge research insights with contact data for better context
    research_lookup = {r["name"]: r["insights"] for r in research_data}
    
    prompt = (
        f"MISSION: Write highly personalized, response-driving outreach emails for each contact below.\n\n"
        f"SENDER CONTEXT:\n"
        f"Name: {sender_name}\n"
        f"Company: {sender_company}\n"
        f"Offering: {offering_desc}\n"
        f"Calendar: {calendar_link or 'Request for calendar link in email'}\n\n"
        f"CONTACTS & RESEARCH:\n{json.dumps(contacts_data, indent=2)}\n\n"
        f"RESEARCH INSIGHTS:\n{json.dumps(research_data, indent=2)}\n\n"
        f"EMAIL REQUIREMENTS:\n"
        f"- Use specific research insights (not generic industry observations)\n"
        f"- Connect insights directly to value proposition\n"
        f"- Personalize for recipient's role and likely priorities\n"
        f"- Include compelling subject line\n"
        f"- Clear, specific call-to-action\n\n"
        f"Each email should feel individually researched and written, not templated.\n\n"
        f"Return format: {{\"emails\": [{{\"company\": \"Company\", \"contact\": \"Contact Name\", \"subject\": \"Subject\", \"body\": \"Email body\", \"personalization_used\": \"What insight was used\"}}]}}"
    )
    resp = agent.run(prompt)
    data = extract_json_or_raise(str(resp.content))
    return data.get("emails", [])


def run_pipeline(target_desc: str, offering_desc: str, sender_name: str, sender_company: str, 
                calendar_link: Optional[str], num_companies: int, email_style: str):
    """Run the complete outreach pipeline with improved error handling and progress tracking."""
    
    # Initialize agents
    company_agent = create_company_finder_agent()
    contact_agent = create_contact_finder_agent()
    phone_agent = create_phone_finder_agent()
    research_agent = create_research_agent()
    email_agent = create_email_writer_agent(email_style)
    
    results = {
        "companies": [],
        "contacts": [],
        "phones": [],
        "research": [],
        "emails": []
    }
    
    try:
        # Step 1: Find companies
        results["companies"] = run_company_finder(
            company_agent, target_desc, offering_desc, num_companies
        )
        
        if results["companies"]:
            # Step 2: Find contacts
            results["contacts"] = run_contact_finder(
                contact_agent, results["companies"], target_desc, offering_desc
            )
            
            if results["contacts"]:
                # Step 3: Find phone numbers (optional, may have low success rate)
                try:
                    results["phones"] = run_phone_finder(phone_agent, results["contacts"])
                except Exception as e:
                    st.warning(f"Phone finding partially failed: {str(e)[:100]}...")
                    results["phones"] = []
                
                # Step 4: Research insights
                results["research"] = run_research(research_agent, results["companies"])
                
                # Step 5: Generate emails
                results["emails"] = run_email_writer(
                    email_agent, results["contacts"], results["research"],
                    offering_desc, sender_name, sender_company, calendar_link
                )
    
    except Exception as e:
        st.error(f"Pipeline error: {str(e)}")
        raise
    
    return results


def main() -> None:
    st.set_page_config(page_title="GTM B2B Outreach - Multi-Agent Pipeline", layout="wide")

    # Sidebar: API keys and settings
    st.sidebar.header("🔑 API Configuration")
    openai_key = st.sidebar.text_input("OpenAI API Key", type="password", value=os.getenv("OPENAI_API_KEY", ""))
    exa_key = st.sidebar.text_input("Exa API Key", type="password", value=os.getenv("EXA_API_KEY", ""))
    
    if openai_key:
        os.environ["OPENAI_API_KEY"] = openai_key
    if exa_key:
        os.environ["EXA_API_KEY"] = exa_key

    if not openai_key or not exa_key:
        st.sidebar.warning("⚠️ Enter both API keys to enable the app")
        st.sidebar.info("Get OpenAI key from: https://platform.openai.com/api-keys")
        st.sidebar.info("Get Exa key from: https://exa.ai/")

    # Main interface
    st.title("🎯 GTM B2B Outreach Multi-Agent Pipeline")
    st.markdown("""
    **Automated B2B prospecting and personalized outreach generation**
    
    This system uses GPT-5 and specialized agents to:
    1. 🔍 Find high-quality target companies
    2. 👥 Identify key decision makers with contact info
    3. 📞 Locate phone numbers (when available)
    4. 🔬 Research company insights and growth signals
    5. ✉️ Generate personalized outreach emails
    """)

    # Input form
    with st.form("outreach_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🎯 Targeting")
            target_desc = st.text_area(
                "Target companies (industry, size, region, tech stack, etc.)",
                height=120,
                placeholder="Example: B2B SaaS companies, 100-1000 employees, using Salesforce, focused on customer success, venture-backed..."
            )
            
            offering_desc = st.text_area(
                "Your product/service offering", 
                height=120,
                placeholder="Example: AI-powered sales coaching platform that helps sales teams improve conversion rates through real-time call analysis and personalized feedback..."
            )
        
        with col2:
            st.subheader("✉️ Outreach Settings")
            sender_name = st.text_input("Your name", value="", placeholder="John Smith")
            sender_company = st.text_input("Your company", value="", placeholder="Acme Solutions")
            calendar_link = st.text_input(
                "Calendar link (optional)", 
                value="", 
                placeholder="https://calendly.com/yourname"
            )
            
            num_companies = st.number_input(
                "Number of companies to find", 
                min_value=1, max_value=10, value=5,
                help="More companies = longer processing time"
            )
            
            email_style = st.selectbox(
                "Email style",
                options=["Professional", "Casual", "Cold", "Consultative"],
                index=0,
                help="Choose the tone and approach for generated emails"
            )

        submitted = st.form_submit_button("🚀 Start Outreach Pipeline", type="primary")

    # Process the pipeline
    if submitted:
        # Validation
        if not openai_key or not exa_key:
            st.error("❌ Please provide both API keys in the sidebar")
        elif not target_desc.strip() or not offering_desc.strip():
            st.error("❌ Please fill in target companies and offering description")
        elif not sender_name.strip():
            st.error("❌ Please provide your name")
        else:
            # Run pipeline with progress tracking
            progress_container = st.container()
            with progress_container:
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                try:
                    # Step 1: Companies
                    status_text.info("🔍 Step 1/5: Finding target companies...")
                    company_agent = create_company_finder_agent()
                    companies = run_company_finder(
                        company_agent, target_desc.strip(), offering_desc.strip(), num_companies
                    )
                    progress_bar.progress(20)
                    st.success(f"✅ Found {len(companies)} companies")

                    if companies:
                        # Step 2: Contacts
                        status_text.info("👥 Step 2/5: Finding decision makers and contact info...")
                        contact_agent = create_contact_finder_agent()
                        contacts_data = run_contact_finder(
                            contact_agent, companies, target_desc.strip(), offering_desc.strip()
                        )
                        progress_bar.progress(40)
                        total_contacts = sum(len(c.get("contacts", [])) for c in contacts_data)
                        st.success(f"✅ Found {total_contacts} contacts across {len(contacts_data)} companies")

                        if contacts_data:
                            # Step 3: Phone numbers (optional)
                            status_text.info("📞 Step 3/5: Finding phone numbers...")
                            phone_agent = create_phone_finder_agent()
                            try:
                                phone_data = run_phone_finder(phone_agent, contacts_data)
                                total_phones = sum(len(c.get("contacts", [])) for c in phone_data)
                                st.success(f"✅ Found {total_phones} phone numbers")
                            except Exception as e:
                                phone_data = []
                                st.warning(f"⚠️ Phone finding had limited success: {str(e)[:100]}...")
                            progress_bar.progress(60)

                            # Step 4: Research
                            status_text.info("🔬 Step 4/5: Researching company insights...")
                            research_agent = create_research_agent()
                            research_data = run_research(research_agent, companies)
                            progress_bar.progress(80)
                            total_insights = sum(len(r.get("insights", [])) for r in research_data)
                            st.success(f"✅ Gathered {total_insights} research insights")

                            # Step 5: Email generation
                            status_text.info("✉️ Step 5/5: Writing personalized outreach emails...")
                            email_agent = create_email_writer_agent(email_style)
                            emails = run_email_writer(
                                email_agent, contacts_data, research_data,
                                offering_desc.strip(), sender_name.strip(), 
                                sender_company.strip(), calendar_link.strip() or None
                            )
                            progress_bar.progress(100)
                            st.success(f"✅ Generated {len(emails)} personalized emails")

                            # Store results
                            st.session_state["gtm_results"] = {
                                "companies": companies,
                                "contacts": contacts_data,
                                "phones": phone_data,
                                "research": research_data,
                                "emails": emails,
                            }
                            status_text.success("🎉 Pipeline completed successfully!")
                        else:
                            st.error("❌ No contacts found for the target companies")
                    else:
                        st.error("❌ No matching companies found. Try adjusting your targeting criteria.")

                except Exception as e:
                    progress_bar.progress(0)
                    status_text.error("❌ Pipeline failed")
                    st.error(f"Error: {str(e)}")
                    st.info("💡 Try simplifying your targeting criteria or check your API keys")

    # Display results
    results = st.session_state.get("gtm_results")
    if results:
        st.divider()
        st.header("📊 Results")
        
        # Summary metrics
        companies = results.get("companies", [])
        contacts = results.get("contacts", [])
        phones = results.get("phones", [])
        research = results.get("research", [])
        emails = results.get("emails", [])
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Companies", len(companies))
        with col2:
            total_contacts = sum(len(c.get("contacts", [])) for c in contacts)
            st.metric("Contacts", total_contacts)
        with col3:
            total_phones = sum(len(c.get("contacts", [])) for c in phones)
            st.metric("Phone Numbers", total_phones)
        with col4:
            st.metric("Emails Generated", len(emails))

        # Detailed results in tabs
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "🏢 Companies", "👥 Contacts", "📞 Phone Numbers", "🔬 Research", "✉️ Emails"
        ])
        
        with tab1:
            st.subheader("Target Companies Found")
            if companies:
                for idx, company in enumerate(companies, 1):
                    with st.expander(f"{idx}. {company.get('name', 'Unknown Company')}", expanded=False):
                        st.write(f"**Website:** {company.get('website', 'N/A')}")
                        st.write(f"**Employee Count:** {company.get('employee_count', 'Unknown')}")
                        st.write(f"**Why it's a fit:** {company.get('why_fit', 'N/A')}")
                        
                        growth_signals = company.get('growth_signals', [])
                        if growth_signals:
                            st.write("**Growth Signals:**")
                            for signal in growth_signals:
                                st.write(f"• {signal}")
            else:
                st.info("No companies found")

        with tab2:
            st.subheader("Decision Makers & Contact Information")
            if contacts:
                for company_data in contacts:
                    company_name = company_data.get('name', 'Unknown Company')
                    company_contacts = company_data.get('contacts', [])
                    
                    if company_contacts:
                        st.markdown(f"### {company_name}")
                        for contact in company_contacts:
                            col1, col2, col3 = st.columns([2, 2, 1])
                            with col1:
                                st.write(f"**{contact.get('full_name', 'Unknown')}**")
                                st.write(contact.get('title', 'Unknown Title'))
                            with col2:
                                email = contact.get('email', 'Not found')
                                inferred_badge = " 🔍" if contact.get('inferred') else " ✅"
                                st.write(f"📧 {email}{inferred_badge}")
                                st.write(f"🔗 {contact.get('source', 'Unknown source')}")
                            with col3:
                                activity = contact.get('last_activity', 'Unknown')
                                st.write(f"📅 {activity}")
                        st.divider()
            else:
                st.info("No contacts found")

        with tab3:
            st.subheader("Phone Numbers")
            if phones and any(c.get('contacts') for c in phones):
                for company_data in phones:
                    company_name = company_data.get('name', 'Unknown Company')
                    company_phones = company_data.get('contacts', [])
                    
                    if company_phones:
                        st.markdown(f"### {company_name}")
                        for contact in company_phones:
                            col1, col2, col3 = st.columns([2, 2, 1])
                            with col1:
                                st.write(f"**{contact.get('full_name', 'Unknown')}**")
                            with col2:
                                phone = contact.get('phone_number', 'Not found')
                                phone_type = contact.get('phone_type', 'unknown')
                                verified_badge = " ✅" if contact.get('verified') else " ~"
                                st.write(f"📞 {phone} ({phone_type}){verified_badge}")
                            with col3:
                                source = contact.get('source', 'Unknown')
                                st.write(f"🔗 {source}")
                        st.divider()
            else:
                st.info("No phone numbers found - this is normal as phone numbers are harder to locate publicly")

        with tab4:
            st.subheader("Company Research Insights")
            if research:
                for company_research in research:
                    company_name = company_research.get('name', 'Unknown Company')
                    insights = company_research.get('insights', [])
                    
                    if insights:
                        with st.expander(f"🔍 {company_name} Insights", expanded=False):
                            for i, insight in enumerate(insights, 1):
                                st.write(f"**{i}.** {insight}")
            else:
                st.info("No research insights gathered")

        with tab5:
            st.subheader("Personalized Outreach Emails")
            if emails:
                for idx, email in enumerate(emails, 1):
                    company = email.get('company', 'Unknown Company')
                    contact = email.get('contact', 'Unknown Contact')
                    
                    with st.expander(f"✉️ {idx}. {company} → {contact}", expanded=False):
                        # Subject line
                        st.markdown(f"**Subject:** `{email.get('subject', 'No subject')}`")
                        st.divider()
                        
                        # Email body
                        body = email.get('body', 'No email content')
                        st.markdown("**Email Body:**")
                        st.text(body)
                        
                        # Personalization note
                        if 'personalization_used' in email:
                            st.divider()
                            st.markdown(f"**Personalization:** {email['personalization_used']}")
                        
                        # Copy button simulation
                        st.divider()
                        st.code(f"Subject: {email.get('subject', '')}\n\n{body}", language=None)
            else:
                st.info("No emails generated")

        # Export options
        if emails:
            st.divider()
            st.subheader("📥 Export Options")
            
            col1, col2 = st.columns(2)
            with col1:
                # CSV export
                import pandas as pd
                
                # Prepare data for CSV
                csv_data = []
                for email in emails:
                    csv_data.append({
                        'Company': email.get('company', ''),
                        'Contact': email.get('contact', ''),
                        'Subject': email.get('subject', ''),
                        'Body': email.get('body', ''),
                        'Personalization': email.get('personalization_used', '')
                    })
                
                df = pd.DataFrame(csv_data)
                csv_buffer = df.to_csv(index=False)
                
                st.download_button(
                    label="📊 Download as CSV",
                    data=csv_buffer,
                    file_name=f"outreach_emails_{sender_company.replace(' ', '_')}.csv",
                    mime="text/csv"
                )
            
            with col2:
                # JSON export
                json_data = {
                    "campaign_info": {
                        "sender_name": sender_name,
                        "sender_company": sender_company,
                        "target_description": target_desc[:200] + "..." if len(target_desc) > 200 else target_desc,
                        "offering_description": offering_desc[:200] + "..." if len(offering_desc) > 200 else offering_desc,
                        "email_style": email_style,
                        "generated_at": str(pd.Timestamp.now())
                    },
                    "results": results
                }
                
                st.download_button(
                    label="💾 Download as JSON",
                    data=json.dumps(json_data, indent=2),
                    file_name=f"outreach_campaign_{sender_company.replace(' ', '_')}.json",
                    mime="application/json"
                )

    # Footer with tips
    st.divider()
    st.markdown("""
    ### 💡 Tips for Better Results
    
    **Targeting:**
    - Be specific about company characteristics (industry, size, tech stack)
    - Include growth indicators (funding stage, recent hiring, expansion)
    - Mention specific pain points your solution addresses
    
    **Offering:**
    - Focus on specific outcomes and benefits
    - Include quantifiable value (time saved, revenue increased, etc.)
    - Mention relevant integrations or technical requirements
    
    **Email Style:**
    - **Professional:** Best for enterprise prospects and formal industries
    - **Casual:** Good for tech/startup companies and younger contacts  
    - **Cold:** Optimized for high open/response rates with strangers
    - **Consultative:** Best when you have strong domain expertise to share
    """)


if __name__ == "__main__":
    main()